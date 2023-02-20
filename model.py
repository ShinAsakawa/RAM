import torch
import torch.nn as nn
import torch.nn.functional as F
#from .utils import check_vals_performance
#from .utils import convert_ids2tensor
#from .utils import timeSince

from termcolor import colored
import random
import numpy as np
import time
import sys

device="cuda:0" if torch.cuda.is_available() else "cpu"

# encoder, decoder の再定義
# エンコーダの中間層の値を計算:= 注意の重み (attn_weights) * エンコーダの出力 (hiden) = 注意を描けた
# この attn_weights.unsqueeze(0) は 第 1 次元が batch になっているようだ

import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    """RNNによる符号化器"""
    def __init__(self,
            n_inp:int=0,
            n_hid:int=0):
        super().__init__()
        self.n_hid = n_hid
        self.n_inp = n_inp

        self.embedding = nn.Embedding(num_embeddings=n_inp, embedding_dim=n_hid)
        self.gru = nn.GRU(input_size=n_hid, hidden_size=n_hid)

    def forward(self,
                inp:torch.Tensor=0,
                hid:torch.Tensor=0,
                device=device
               ):
        embedded = self.embedding(inp).view(1, 1, -1)
        out = embedded
        out, hid = self.gru(out, hid)
        return out, hid

    def initHidden(self)->torch.Tensor:
        return torch.zeros(1, 1, self.n_hid, device=device)


class AttnDecoderRNN(nn.Module):
    """注意付き復号化器の定義"""
    def __init__(self,
                 n_hid:int=0,
                 n_out:int=0,
                 dropout_p:float=0.0,
                 max_length:int=0):
        super().__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.dropout_p = dropout_p
        self.max_length = max_length

        # n_out と n_inp は同じ特徴数であるから，n_out が 入力特徴数となる
        self.embedding = nn.Embedding(num_embeddings=n_out, embedding_dim=n_hid)

        # n_hid + n_hid -> max_length の線形層を定義
        self.attn = nn.Linear(in_features=n_hid * 2, out_features=max_length)

        # n_hid + n_hid -> n_hid の線形層
        self.attn_combine = nn.Linear(in_features=n_hid * 2, out_features=n_hid)

        self.dropout = nn.Dropout(p=dropout_p)

        # GRU には batch_first オプションをつけていない。これは，一系列ずつしか処理しないため
        self.gru = nn.GRU(input_size=n_hid, hidden_size=n_hid)

        # 最終出力
        self.out_layer = nn.Linear(in_features=n_hid, out_features=n_out)

    def forward(self,
                inp:torch.Tensor=None,  # 旧版では int だが正しくは torch.Tensor
                hid:torch.Tensor=None,  # 旧版では int だが正しくは torch.Tensor
                encoder_outputs:torch.Tensor=None,
                device:torch.device=device):
        embedded = self.embedding(inp).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # 注意の重みを計算
        # 入力 (embedded[0]) と 中間層 (hidden[0]) とを第 2 次元に沿って連結 (torch.cat(dim=1))
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hid[0]), 1)), dim=1)

        # エンコーダの中間層の値を計算:= 注意の重み (attn_weights) * エンコーダの出力 (hiden) = 注意を描けた
        # この attn_weights.unsqueeze(0) は 第 1 次元が batch になっているようだ
        # print(f'attn_weights.unsqueeze(0).size():{attn_weights.unsqueeze(0).size()}')
        # print(f'encoder_outputs.unsqueeze(0).size():{encoder_outputs.unsqueeze(0).size()}')
        #sys.exit()
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # print(f'attn_weight.unsqeeze(0).size():{attn_weights.unsqueeze(0).size()}')
        # print(f'encoder_outputs.unsqeeze(0).size():{encoder_outputs.unsqueeze(0).size()}')
        # print(f'attn_applied.size():{attn_applied.size()}')
        out = torch.cat((embedded[0], attn_applied[0]), 1)
        # print(f'out.size():{out.size()}')
        out = self.attn_combine(out).unsqueeze(0)
        # print(f'out.size():{out.size()}')
        # sys.exit()

        out = F.relu(out)
        out, hid = self.gru(out, hid)

        out = F.log_softmax(self.out_layer(out[0]), dim=1)
        return out, hid, attn_weights

    def initHidden(self)->torch.Tensor:
        return torch.zeros(1, 1, self.n_hid, device=device)


def call_one_seq2seq(
    inp_tensor:torch.Tensor=None,
    tgt_tensor:torch.Tensor=None,
    encoder:torch.nn.Module=None,
    decoder:torch.nn.Module=None,
    encoder_optimizer:torch.optim=torch.optim, #.adam.Adam,
    decoder_optimizer:torch.optim=torch.optim, # .adam.Adam,
    criterion:torch.nn.modules.loss=torch.nn.NLLLoss(),
    #max_length:int=1,
    target_vocab:list=None,
    is_teacher_forcing:bool=False,
    device:torch.device=device)->float:
    """一対の入出力系列 `input_tensor` と `target_tensor` (いずれも torch.Tensor) を受け取r，encoder と decoder を呼び出す関数
    ただし teacher_forcing は，この関数の外部で与えられると仮定
    max_length の自動生成を試してみる
    """

    enc_hid = encoder.initHidden()    # 符号化器の中間層を初期化
    encoder_optimizer.zero_grad()     # 符号化器の最適化関数の初期化
    decoder_optimizer.zero_grad()     # 復号化器の最適化関数の初期化
    inp_len = inp_tensor.size(0)      # 0 次元目が系列であることを仮定
    tgt_len = tgt_tensor.size(0)

    # enc_outputs の定義。max_length は系列の最大長。注意機構では，この enc_outputs を使う
    enc_outputs = torch.zeros(inp_len, encoder.n_hid, device=device)

    if inp_len > enc_outputs.size(0):
        print(f'inp_len:{inp_len}, enc_outputs.size():{enc_outputs.size()}')
        print(f'inp_tensor:{inp_tensor}')
        sys.exit()
    loss = 0.                          # 損失値
    for ei in range(inp_len):          # 入力時刻分だけ反復
        enc_out, enc_hid = encoder(inp=inp_tensor[ei], hid=enc_hid, device=device)
        enc_outputs[ei] = enc_out[0, 0]

    # 復号化器の最初の入力データとして <SOW> (単語の開始を示す特殊トークン) を代入
    dec_inp = torch.tensor([[target_vocab.index('<SOW>')]], device=device)
    dec_hid = enc_hid                  # 復号化器の中間層の初期値を符号化器の最終時刻の中間層とする

    ok_flag = True
    if is_teacher_forcing:                     # 教師強制をする場合: 次時刻の入力を強制的にターゲット (正解) とする
        for di in range(tgt_len):
            dec_out, dec_hid, dec_attn = decoder(
                dec_inp, dec_hid, enc_outputs, device=device)
            dec_inp = tgt_tensor[di]           # 教師強制

            loss += criterion(dec_out, tgt_tensor[di])
            ok_flag = (ok_flag) and (dec_out.argmax() == tgt_tensor[di].detach().cpu().numpy()[0])
            if dec_inp.item() == target_vocab.index('<EOW>'):
                break
    else:                                       # 教師強制をしない場合: 自身の出力を次刻の入力とする
        for di in range(tgt_len):
            dec_out, dec_hid, dec_attn = decoder(
                dec_inp,   dec_hid, enc_outputs, device=device)
            topv, topi = dec_out.topk(1)        # 教師強制しない
            dec_inp = topi.squeeze().detach()   # detach() しないと bptt をリカーシブにしてしまう

            loss += criterion(dec_out, tgt_tensor[di])
            ok_flag = (ok_flag) and (dec_out.argmax() == tgt_tensor[di].detach().cpu().numpy()[0])
            if dec_inp.item() == target_vocab.index('<EOW>'):
                break

    #loss.backward()           # 誤差逆伝播
    #encoder_optimizer.step()  # encoder の学習
    #decoder_optimizer.step()  # decoder の学習
    return loss.item() / tgt_len, ok_flag


def train_one_seq2seq(
    inp_tensor:torch.Tensor=None,
    tgt_tensor:torch.Tensor=None,
    encoder:torch.nn.Module=None,
    decoder:torch.nn.Module=None,
    encoder_optimizer:torch.optim=torch.optim, #.adam.Adam,
    decoder_optimizer:torch.optim=torch.optim, # .adam.Adam,
    criterion:torch.nn.modules.loss=torch.nn.NLLLoss(),
    max_length:int=1,
    target_vocab:list=None,
    teacher_forcing_ratio:float=0.,
    device:torch.device=device)->float:

    """一対の系列 `input_tensor` と `target_tensor` (いずれも torch.Tensor) を受け取って，
    encoder と decoder の訓練を行う関数
    """

    enc_hid = encoder.initHidden()    # 符号化器の中間層を初期化
    encoder_optimizer.zero_grad()     # 符号化器の最適化関数の初期化
    decoder_optimizer.zero_grad()     # 復号化器の最適化関数の初期化
    inp_len = inp_tensor.size(0)      # 0 次元目が系列であることを仮定
    tgt_len = tgt_tensor.size(0)

    # enc_outputs の定義。max_length は系列の最大長。注意機構では，この enc_outputs を使う
    enc_outputs = torch.zeros(max_length, encoder.n_hid, device=device)

    if inp_len > enc_outputs.size(0):
        print(f'inp_len:{inp_len}, enc_outputs.size():{enc_outputs.size()}')
        print(f'inp_tensor:{inp_tensor}')
        sys.exit()
    loss = 0.                          # 損失値
    for ei in range(inp_len):          # 入力時刻分だけ反復
        enc_out, enc_hid = encoder(inp=inp_tensor[ei], hid=enc_hid, device=device)
        enc_outputs[ei] = enc_out[0, 0]

    # 復号化器の最初の入力データとして <SOW> (単語の開始を示す特殊トークン) を代入
    dec_inp = torch.tensor([[target_vocab.index('<SOW>')]], device=device)
    dec_hid = enc_hid                 # 復号化器の中間層の初期値を符号化器の最終時刻の中間層とする

    ok_flag = True

    # ｢教師強制｣ をするか否かを確率的に決める
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:                    # 教師強制をする場合: 次時刻の入力を強制的にターゲット (正解) とする
        for di in range(tgt_len):
            dec_out, dec_hid, dec_attn = decoder(
                dec_inp, dec_hid, enc_outputs, device=device)
            dec_inp = tgt_tensor[di]           # 教師強制

            loss += criterion(dec_out, tgt_tensor[di])
            ok_flag = (ok_flag) and (dec_out.argmax() == tgt_tensor[di].detach().cpu().numpy()[0])
            if dec_inp.item() == target_vocab.index('<EOW>'):
                break

    else:                                       # 教師強制をしない場合: 自身の出力を次刻の入力とする
        for di in range(tgt_len):
            dec_out, dec_hid, dec_attn = decoder(
                dec_inp,   dec_hid, enc_outputs, device=device)
            topv, topi = dec_out.topk(1)        # 教師強制しない
            dec_inp = topi.squeeze().detach()   # detach() しないと bptt をリカーシブにしてしまう

            loss += criterion(dec_out, tgt_tensor[di])
            ok_flag = (ok_flag) and (dec_out.argmax() == tgt_tensor[di].detach().cpu().numpy()[0])
            if dec_inp.item() == target_vocab.index('<EOW>'):
                break

    loss.backward()           # 誤差逆伝播
    encoder_optimizer.step()  # encoder の学習
    decoder_optimizer.step()  # decoder の学習
    return loss.item() / tgt_len, ok_flag


from .utils import *

def train_epochs(
    epochs:int=1,
    encoder:torch.nn.Module=None,
    decoder:torch.nn.Module=None,
    encoder_optimizer:torch.optim=torch.optim, #.adam.Adam,
    decoder_optimizer:torch.optim=torch.optim, #.adam.Adam,
    criterion:torch.nn.modules=torch.nn.NLLLoss(),
    source_vocab:list=None, target_vocab:list=None,
    source_ids:str=None, target_ids:list=None,
    params:dict=None,
    lr:float=0.0001,
    #n_sample:int=3,  # 削除予定。evaluateRandomly に使っていた。
    teacher_forcing_ratio=False,
    train_dataset:torch.utils.data.Dataset=None,
    val_dataset:dict=None,
    max_length:int=1,
    device=device)->list:
    '''`train_one_seq2seq()` を反復して呼び出してモデルを学習させる'''

    start_time = time.time()

    #criterion = params['loss_func']

    losses = []
    for epoch in range(epochs):

        if val_dataset != None:
            encoder.eval(); decoder.eval()
            _val = check_vals_performance(
                _dataset=val_dataset,
                encoder=encoder,decoder=decoder,
                source_vocab=source_vocab, target_vocab=target_vocab,
                max_length=max_length)
        # if n_sample > 0:
        #     evaluateRandomly(encoder, decoder, n=n_sample)

        encoder.train(); decoder.train()
        epoch_loss, ok_count = 0, 0
        #エポックごとに学習順をシャッフル
        learning_order = np.random.permutation(train_dataset.__len__())
        for i in range(train_dataset.__len__()):
            idx = learning_order[i]
            inp_ids, tgt_ids = train_dataset.__getitem__(idx)
            inp_tensor = convert_ids2tensor(inp_ids)
            tgt_tensor = convert_ids2tensor(tgt_ids)
            loss, ok_flag = train_one_seq2seq(
                inp_tensor=inp_tensor, tgt_tensor=tgt_tensor,
                encoder=encoder, decoder=decoder,
                encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer,
                criterion=criterion,
                max_length=max_length,
                target_vocab=target_vocab,
                teacher_forcing_ratio=teacher_forcing_ratio, device=device)

            epoch_loss += loss
            ok_count += 1 if ok_flag else 0

        losses.append(epoch_loss/train_dataset.__len__())
        print(colored(f'エポック:{epoch:2d} 損失:{epoch_loss/train_dataset.__len__():.2f}', 'cyan', attrs=['bold']),
              colored(f'{timeSince(start_time, (epoch+1) * train_dataset.__len__()/(epochs * train_dataset.__len__()))}',
                      'cyan', attrs=['bold']),
              colored(f'訓練データ精度:{ok_count/train_dataset.__len__():.3f}', 'cyan', attrs=['bold']),
              colored(f'検証データ:{_val}', 'cyan', attrs=['bold'])
              )


    return losses


def train_epochs_with_config(configs:dict=None, verbose=False):
    '''`train_one_seq2seq()` を反復して呼び出してモデルを学習させる'''

    epochs = configs.get('epochs', 1)
    losses = configs.get('losses', [])
    encoder = configs.get('encoder')
    decoder = configs.get('decoder')
    encoder_optimizer = configs.get('encoder_optimizer')
    decoder_optimizer = configs.get('decoder_optimizer')
    max_length = configs.get('max_length') 
    print(f'max_length:{max_length}')
    val_dataset = configs.get('val_dataset')
    #val_dataset = configs.get('val_dataset', None)
    train_dataset = configs.get('train_dataset', None)
    teacher_forcing_ratio = configs.get('teacher_forcing_ratio')
    source_vocab = configs.get('source_vocab')
    target_vocab = configs.get('target_vocab')
    target_list = configs.get('target_list')
    device = configs.get('device')
    criterion = configs.get('loss_func')
    lr = configs.get('lr')

    perfs = configs.get('perfs', {})
    losses = perfs.get('losses', [])
    train_accuracy = perfs.get('train_accuarcy', [])
    val_accuracy = perfs.get('val_accuracy', [])

    start_time = time.time()
    color = 'cyan'


    if verbose:
        for k, v in sorted(configs.items(), key=lambda i: i[0]):
            print(f'{k}:{colored(v, color=color, attrs=["bold"])}')

    for epoch in range(epochs):

        if (val_dataset != None) and (len(val_dataset) > 0):
            encoder.eval(); decoder.eval()
            _val = check_a_dataset(
                _dataset=val_dataset,
                encoder=encoder,decoder=decoder,
                source_vocab=source_vocab, target_vocab=target_vocab,
                max_length=max_length)

        encoder.train(); decoder.train()
        epoch_loss, ok_count = 0, 0

        #エポックごとに学習順をシャッフル
        learning_order = np.random.permutation(train_dataset.__len__())
        for i in range(train_dataset.__len__()):
            idx = learning_order[i]
            inp_ids, tgt_ids = train_dataset.__getitem__(idx)
            inp_tensor = convert_ids2tensor(inp_ids)
            tgt_tensor = convert_ids2tensor(tgt_ids)
            loss, ok_flag = train_one_seq2seq(
                inp_tensor=inp_tensor, tgt_tensor=tgt_tensor,
                encoder=encoder, decoder=decoder,
                encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer,
                criterion=criterion,
                max_length=max_length,
                target_vocab=target_vocab,
                teacher_forcing_ratio=teacher_forcing_ratio, device=device)

            epoch_loss += loss
            ok_count += 1 if ok_flag else 0

        losses.append(epoch_loss/train_dataset.__len__())
        _train_accuracy = ok_count/train_dataset.__len__()
        train_accuracy.append(_train_accuracy)
        if (val_dataset != None) and (len(val_dataset) > 0):
            _val_accuracy   = _val
            val_accuracy.append(_val_accuracy[0])
            print(
                colored(f'エポック:{epoch:2d} 損失:{epoch_loss/train_dataset.__len__():.2f}', color=color, attrs=['bold']),
                colored(f'{timeSince(start_time, (epoch+1) * train_dataset.__len__()/(epochs * train_dataset.__len__()))}', 
                    color=color, attrs=['bold']),
                colored(f'訓練データ精度:{_train_accuracy:.3f}', color=color, attrs=['bold']),
                colored(f'検証データ:{_val_accuracy}', color=color, attrs=['bold']))
        else:
            print(
                colored(f'エポック:{epoch:2d} 損失:{epoch_loss/train_dataset.__len__():.2f}', color=color, attrs=['bold']),
                colored(f'{timeSince(start_time, (epoch+1) * train_dataset.__len__()/(epochs * train_dataset.__len__()))}', 
                    color=color, attrs=['bold']),
                colored(f'訓練データ精度:{_train_accuracy:.3f}', color=color, attrs=['bold']))
                #colored(f'訓練データ精度:{ok_count/train_dataset.__len__():.3f}', color=color, attrs=['bold']))

    perfs['losses'] = losses
    perfs['train_accuracy'] = train_accuracy
    perfs['val_accuracy'] = val_accuracy

    return perfs