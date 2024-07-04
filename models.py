import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim

import time
import datetime
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from termcolor import colored

class Seq2Seq_wAtt(nn.Module):
    """ 注意つき符号化器‐復号化器モデル
    Bahdanau, Cho, & Bengio (2015) NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE, arXiv:1409.0473
    """
    def __init__(self,
                 enc_vocab_size:int,
                 dec_vocab_size:int,
                 n_hid:int,
                 n_layers:int=2,
                 bidirectional:bool=False):
        super().__init__()

        # Encoder 側の入力トークン id を多次元ベクトルに変換
        self.encoder_emb = nn.Embedding(num_embeddings=enc_vocab_size,
                                        embedding_dim=n_hid,
                                        padding_idx=0)

        # Decoder 側の入力トークン id を多次元ベクトルに変換
        self.decoder_emb = nn.Embedding(num_embeddings=dec_vocab_size,
                                        embedding_dim=n_hid,
                                        padding_idx=0)

        # Encoder LSTM 本体
        self.encoder = nn.LSTM(input_size=n_hid,
                               hidden_size=n_hid,
                               num_layers=n_layers,
                               batch_first=True,
                               bidirectional=bidirectional)

        # Decoder LSTM 本体
        self.decoder = nn.LSTM(input_size=n_hid,
                               hidden_size=n_hid,
                               num_layers=n_layers,
                               batch_first=True,
                               bidirectional=bidirectional)

        # 文脈ベクトルと出力ベクトルの合成を合成する層
        bi_fact = 2 if bidirectional else 1
        self.combine_layer = nn.Linear(bi_fact * 2 * n_hid, n_hid)

        # 最終出力層
        self.out_layer = nn.Linear(n_hid, dec_vocab_size)

    def forward(self, enc_inp, dec_inp):

        enc_emb = self.encoder_emb(enc_inp)
        enc_out, (hnx, cnx) = self.encoder(enc_emb)

        dec_emb = self.decoder_emb(dec_inp)
        dec_out, (hny, cny) = self.decoder(dec_emb,(hnx, cnx))

        # enc_out は (バッチサイズ，ソースの単語数，中間層の次元数)
        # ソース側 (enc_out) の各単語とターゲット側 (dec_out) の各単語との類似度を測定するため
        # 両テンソルの内積をとるため ソース側 (enc_out) の軸を入れ替え
        enc_outP = enc_out.permute(0,2,1)

        # sim の形状は (バッチサイズ, 中間層の次元数，ソースの単語数)
        sim = torch.bmm(dec_out, enc_outP)

        # sim の各次元のサイズを記録
        batch_size, dec_word_size, enc_word_size = sim.shape

        # sim に対して，ソフトマックスを行うため形状を変更
        simP = sim.reshape(batch_size * dec_word_size, enc_word_size)

        # simP のソフトマックスを用いて注意の重み alpha を算出
        alpha = F.softmax(simP,dim=1).reshape(batch_size, dec_word_size, enc_word_size)

        # 注意の重み alpha に encoder の出力を乗じて，文脈ベクトル c_t とする
        c_t = torch.bmm(alpha, enc_out)

        # torch.cat だから c_t と dec_out とで合成
        dec_out_ = torch.cat([c_t, dec_out], dim=2)
        dec_out_ = self.combine_layer(dec_out_)

        return self.out_layer(dec_out_)


# # 以下確認作業
# ds = psylex71_ds_o2p
# o2p = Seq2Seq_wAtt(enc_vocab_size=len(ds.grapheme),
#                    dec_vocab_size=len(ds.phoneme),
#                    n_layers=n_layers,
#                    bidirectional=bidirectional,
#                    n_hid=n_hid).to(device)
# print(o2p.eval())

class Seq2Seq(nn.Module):
    """ 簡易符号化器‐復号化器モデル """
    def __init__(self,
                 enc_vocab_size:int,
                 dec_vocab_size:int,
                 n_hid:int,
                 n_layers:int=2,
                 bidirectional:bool=False):
        super().__init__()

        # Encoder 側の入力トークン id を多次元ベクトルに変換
        self.encoder_emb = nn.Embedding(num_embeddings=enc_vocab_size,
                                        embedding_dim=n_hid,
                                        padding_idx=0)

        # Decoder 側の入力トークン id を多次元ベクトルに変換
        self.decoder_emb = nn.Embedding(num_embeddings=dec_vocab_size,
                                        embedding_dim=n_hid,
                                        padding_idx=0)

        # Encoder LSTM 本体
        self.encoder = nn.LSTM(input_size=n_hid,
                               hidden_size=n_hid,
                               num_layers=n_layers,
                               batch_first=True,
                               bidirectional=bidirectional)

        # Decoder LSTM 本体
        self.decoder = nn.LSTM(input_size=n_hid,
                               hidden_size=n_hid,
                               num_layers=n_layers,
                               batch_first=True,
                               bidirectional=bidirectional)

        # 文脈ベクトルと出力ベクトルの合成を合成する層
        bi_fact = 2 if bidirectional else 1
        self.combine_layer = nn.Linear(bi_fact * 2 * n_hid, n_hid)

        # 最終出力層
        self.out_layer = nn.Linear(n_hid, dec_vocab_size)

    def forward(self, enc_inp, dec_inp):

        enc_emb = self.encoder_emb(enc_inp)
        enc_out, (hnx, cnx) = self.encoder(enc_emb)

        dec_emb = self.decoder_emb(dec_inp)
        dec_out, (hny, cny) = self.decoder(dec_emb,(hnx, cnx))

        return self.out_layer(dec_out)

        # # enc_out は (バッチサイズ，ソースの単語数，中間層の次元数)
        # # ソース側 (enc_out) の各単語とターゲット側 (dec_out) の各単語との類似度を測定するため
        # # 両テンソルの内積をとるため ソース側 (enc_out) の軸を入れ替え
        # enc_outP = enc_out.permute(0,2,1)

        # # sim の形状は (バッチサイズ, 中間層の次元数，ソースの単語数)
        # sim = torch.bmm(dec_out, enc_outP)

        # # sim の各次元のサイズを記録
        # batch_size, dec_word_size, enc_word_size = sim.shape

        # # sim に対して，ソフトマックスを行うため形状を変更
        # simP = sim.reshape(batch_size * dec_word_size, enc_word_size)

        # # simP のソフトマックスを用いて注意の重み alpha を算出
        # alpha = F.softmax(simP,dim=1).reshape(batch_size, dec_word_size, enc_word_size)

        # # 注意の重み alpha に encoder の出力を乗じて，文脈ベクトル c_t とする
        # c_t = torch.bmm(alpha, enc_out)

        # # torch.cat だから c_t と dec_out とで合成
        # dec_out_ = torch.cat([c_t, dec_out], dim=2)
        # dec_out_ = self.combine_layer(dec_out_)

        # return self.out_layer(dec_out_)


class Vec2Seq(nn.Module):
    def __init__(self,
                 sem_dim:int,
                 dec_vocab_size:int,
                 n_hid:int,
                 n_layers:int=2,
                 bidirectional:bool=False):
        super().__init__()

        # 単語の意味ベクトル a.k.a 埋め込み表現 を decoder の中間層に接続するための変換層
        # 別解としては，入力層に接続する方法があるが，それはまた別実装にする
        self.enc_transform_layer = nn.Linear(
            in_features=sem_dim,
            out_features=n_hid)
        self.decoder_emb = nn.Embedding(
            num_embeddings=dec_vocab_size,
            embedding_dim=n_hid,
            padding_idx=0)

        self.decoder = nn.LSTM(
            input_size=n_hid,
            hidden_size=n_hid,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional)

        # 最終出力層
        self.bi_fact = 2 if bidirectional else 1
        self.out_layer = nn.Linear(self.bi_fact * n_hid, dec_vocab_size)

    def forward(self, enc_inp, dec_inp):
        enc_emb = self.enc_transform_layer(enc_inp)
        hnx, cnx = enc_emb.clone(), enc_emb.clone()
        hnx = hnx.unsqueeze(0)
        cnx = cnx.unsqueeze(0)

        if self.bi_fact == 2:
            hnx = hnx.repeat(2)
            cnx = cnx.repeat(2)

        dec_emb = self.decoder_emb(dec_inp)

        batch_size = enc_inp.size(0)
        exp_hid_size = self.decoder.get_expected_hidden_size(enc_inp, batch_sizes=[batch_size])
        dec_out, (hny, cny) = self.decoder(dec_emb,(hnx, cnx))

        return self.out_layer(dec_out)

# # 以下確認作業
# ds = psylex71_ds_s2p
# s2p = Vec2Seq(
#     sem_dim=ds.w2v.vector_size,
#     dec_vocab_size=len(ds.phoneme),
#     n_hid=n_hid,
#     n_layers=n_layers,
#     bidirectional=bidirectional).to(device)
# print(s2p.eval())

class Seq2Vec(nn.Module):
    """ 系列データを符号化器に与え，埋め込みデータ (ベクトル) を復号化するモデル
    """
    def __init__(
        self,
        sem_dim:int,
        enc_vocab_size:int,
        n_hid:int,
        n_layers:int=2,
        bidirectional:bool=False):

        super().__init__()
        self.encoder_emb = nn.Embedding(
            num_embeddings=enc_vocab_size,
            embedding_dim=n_hid,
            padding_idx=0)

        self.encoder = nn.LSTM(
            input_size=n_hid,
            hidden_size=n_hid,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional)

        bi_fact = 2 if bidirectional else 1

        self.encoder_out = nn.Linear(
            in_features=n_hid * bi_fact,
            out_features=enc_vocab_size)

        self.out_layer = nn.Linear(
            in_features=n_hid * bi_fact,
            out_features=sem_dim)

    def forward(self, enc_inp):
        enc_emb = self.encoder_emb(enc_inp)
        enc_out, (hid, cel) = self.encoder(enc_emb)
        _enc_out = self.encoder_out(enc_out)
        _sem = self.out_layer(hid)
        return _sem, _enc_out

# # 以下確認作業
# ds = psylex71_ds_o2s
# o2s = Seq2Vec(sem_dim=ds.w2v.vector_size,
#               enc_vocab_size=len(ds.grapheme),
#               n_layers=n_layers,
#               bidirectional=bidirectional,
#               n_hid=n_hid).to(device)
# print(o2s.eval())

class Vec2Vec(nn.Module):
    """ ベクトル埋め込み表現をベクトル埋め込み表現へと変換
    """
    def __init__(self,
                 sem_dim:int,
                 n_hid:int):

        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(sem_dim, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, sem_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# # 以下確認作業
# ds = psylex71_ds_s2s
# s2s = Vec2Vec(sem_dim=ds.w2v.vector_size,
#               n_hid=n_hid).to(device)
# print(s2s.eval())

def _collate_fn(batch):
    inps, tgts = list(zip(*batch))
    inps = list(inps)
    tgts = list(tgts)
    return inps, tgts

import numpy as np

def fit_seq2seq(
    model:torch.nn.modules.module.Module=None,
    #model:torch.nn.modules.module.Module=o2p,
    epochs:int=10,
    ds:Dataset=None,
    #ds:Dataset=psylex71_ds_o2p,
    batch_size=10,
    collate_fn=_collate_fn,
    #dataloader:torch.utils.data.dataloader.DataLoader=dl_o2p,
    optimizer:torch.optim=None,
    criterion:torch.nn.modules.loss=nn.CrossEntropyLoss(ignore_index=-1),
    interval:int=None,
    isPrint:bool=False,
    losses:list=None,
    isDraw:bool=True,
    lr:float=0.001,
    device:str="cpu" ):
    """ Seq2seq の訓練に用いる関数"""

    start_time = time.time()   # 開始時刻の保存

    dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn)


    if losses == None:
        losses = []

    model.train()

    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for _inp, _tch in dataloader:
            enc_inp = pad_sequence(_inp, batch_first=True).to(device)
            dec_inp = pad_sequence(_tch, batch_first=True).to(device)
            tch = pad_sequence(_tch, batch_first=True, padding_value=-1.0).to(device)
            out = model(enc_inp, dec_inp).to(device)
            loss = criterion(out[0], tch[0])
            for h in range(1,len(tch)):
                loss += criterion(out[h], tch[h])
            losses.append(loss.item()/len(_inp))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if isDraw:
        plt.plot(losses)
        plt.title(f'epochs:{epochs}, batch_size:{batch_size}, time collapsed:{total_time_str}')
        #plt.title(f'epochs:{epochs}, batch_size:{batch_size}, n_hid:{n_hid}, n_layers:{n_layers}, time collapsed:{total_time_str}')
        plt.show()

    return {'Training time':total_time_str,
            'losses': losses,
            'optimizer': optimizer,
            'time': total_time
           }


def eval_seq2seq(
    model:torch.nn.modules.module.Module=None,
    ds:Dataset=None,
    # model:torch.nn.modules.module.Module=o2p,
    # ds:Dataset=psylex71_ds_o2p,
    isPrint:bool=False,
    errors:list=None,
    device:str="cpu",
    ):

    model.eval()
    if errors == None:
        errors=[]

    for N in tqdm(range(ds.__len__())):
        x, y = ds.__getitem__(N)
        enc_inp, dec_inp = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        grand_truth = y.detach().numpy()[1:-1]
        y_hat = model(enc_inp, dec_inp).to('cpu')
        y_hat = np.argmax(y_hat.squeeze(0).detach().numpy(), axis=1)[1:-1]

        if len(y_hat) == len(grand_truth):
            n_correct = np.array((y_hat == grand_truth).sum())
            isOK = n_correct == len(grand_truth)
        else:
            isOK = False

        if not isOK:
            wrd = ds.getitem(N)[0]
            _out = ds.target_ids2target(y_hat)
            errors.append((N, wrd, _out, y_hat))
            if isPrint:
                color = 'grey' if isOK else 'red'
                wrd = ds.getitem(N)[0]
                print(colored(f'{N:05d}', color),
                      colored(wrd, color='grey'), # , attrs=["bold"]),
                      colored(y_hat,color,attrs=["bold"]),
                      colored(ds.target_ids2target(y_hat), color, attrs=["bold"]),
                      f'<-{ds.target_ids2target(grand_truth)}')

    cr = len(errors) / N
    return {'エラー':errors,
            '正解率': (1.-cr) * 100}


def fit_seq2vec(
    model:torch.nn.modules.module.Module=None,
    epochs:int=10,
    ds:Dataset=None,
    batch_size=10,
    collate_fn=_collate_fn,
    #dataloader:torch.utils.data.dataloader.DataLoader=dl_o2s,
    optimizer:torch.optim=None,
    lr:float=0.001,
    criterion_enc:torch.nn.modules.loss=nn.CrossEntropyLoss(ignore_index=-1),
    criterion_dec:torch.nn.modules.loss=nn.MSELoss(),
    interval:int=None,
    isPrint:bool=False,
    losses:list=None,
    isDraw:bool=True,
    device:str="cpu"):
    """ Seq2vec の訓練に用いる関数"""

    start_time = time.time()   # 開始時刻の保存

    dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn)

    if losses == None:
        losses = []

    model.train()

    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion_dec = nn.MSELoss()
    criterion_enc = nn.CrossEntropyLoss(ignore_index=-1)
    #criterion_enc = nn.CrossEntropyLoss(ignore_index=0)

    if interval == None:
        interval = int(ds.__len__()/batch_size) >> 2

    for epoch in range(epochs):
        i = 0
        for _inp, _tch in dataloader:
            enc_inp = pad_sequence(_inp, batch_first=True).to(device)
            tch = pad_sequence(_tch, batch_first=True, padding_value=-1.0).to(device)
            out, enc_out = model(enc_inp)

            out = out.squeeze(0)
            loss = criterion_dec(out, tch)
            for _x, _y in zip(enc_out[:,:-1,:], enc_inp[:,1:]):
                loss += criterion_enc(_x, _y)
            losses.append(loss.item()/len(_inp))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            if (i % interval) == 0:
                print(f'epoch:{epoch+1:2d}',
                      f'batch:{i:2d}',
                      f'loss:{loss.item()/batch_size:.5f}')

    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if isDraw:
        plt.plot(losses)
        plt.title(f'epochs:{epochs}, batch_size:{batch_size}, time collapsed:{total_time_str}')
        #plt.title(f'epochs:{epochs}, batch_size:{batch_size}, n_hid:{n_hid}, n_layers:{n_layers}, time collapsed:{total_time_str}')
        plt.show()

    return {'Training time': total_time_str,
            'losses': losses,
            'optimizer': optimizer,
            'time': total_time }

#result = fit_seq2vec(epochs=10, model=p2s, lr=0.0001, ds=psylex71_ds_p2s) # , dataloader=dl_p2s); # 聴理解モデル
#fit_seq2vec(epochs=50, model=o2s, lr=0.0001, ds=psylex71_ds_o2s, dataloader=dl_o2s); # 印字理解モデル
#fit_seq2vec(epochs=20, model=o2s, lr=0.00001, ds=psylex71_ds_o2s, dataloader=dl_o2s); # 印字理解モデル
#fit_seq2vec(epochs=20, model=o2s, lr=0.0001, ds=psylex71_ds_o2s); # 印字理解モデル

import scipy

def eval_seq2vec(
    model:torch.nn.modules.module.Module=None,
    ds:Dataset=None,
    isPrint:bool=False,
    errors:list=None,
    device:str="cpu",
    top_N:int = 3,
    ):

    model.eval()
    if errors == None:
        errors=[]

    for N in tqdm(range(ds.__len__())):
        _inp, _tch = ds.__getitem__(N)

        enc_inp, dec_inp = _inp.unsqueeze(0).to(device), _tch.unsqueeze(0).to(device)

        # モデルに入力して出力を取得する
        out_vec, _ = model(enc_inp)
        out_vec = out_vec.detach().squeeze(0).cpu().numpy()[0]

        grand_truth = ds.getitem(N)  # 正解を得る
        tgt_wrd = grand_truth[0]     # 正解単語を得る

        # 正解単語の意味的類似語を得る。最後に [1:] しているのは自分自身は不要だから
        tgt_neighbors = ds.w2v.most_similar(_tch.detach().cpu().numpy())[1:]

        # モデル出力から得られた単語ベクトルの意味的類似語を得る
        out_neighbors = ds.w2v.similar_by_vector(out_vec)
        #out_neighbors = ds.w2v.similar_by_vector(out_vec.detach().squeeze(0).numpy())
        out_neighbors = ds.w2v.most_similar(out_vec)
        #out_neighbors = ds.w2v.most_similar(out_vec.squeeze(0).detach().numpy())
        #print(out_neighbors_)

        # 正解単語の埋め込みベクトルを得る
        tgt_vec = ds.w2v.get_vector(tgt_wrd)
        tch_vec = _tch.detach().squeeze(0).numpy()

        w2v_cos = ds.w2v.cosine_similarities(out_vec,[tch_vec])
        spy_euc = scipy.spatial.distance.euclidean(out_vec, tch_vec)

        print(f'{N:5d} '
              f'正解:{tgt_wrd},', end=" ")
        print('出力:', end="")
        for _ in out_neighbors[:top_N]:
            print(colored(f'{_[0]}:{_[1]:.3f}',color='blue',attrs=['bold']), end=" ")

        print('<- 正解:', end="")
        for _ in tgt_neighbors[:top_N]:
            print(f'{_[0]}:{_[1]:.3f}', end=" ")
        print(f'出力と正解との距離 cos_sim: {w2v_cos[0]:.3f},'
              f'euc_dist: {spy_euc:.3f}')



import copy

class VecVec2Seq(nn.Module):
    def __init__(
        self,
        inp_dim1:int,           # エンコーダ1 埋め込みベクトル次元
        inp_dim2:int,           # エンコーダ2 埋め込みベクトル次元
        dec_vocab_size:int,     # デコーダ入力次元
        n_hid:int,              # デコーダ中間総数
        decoder:nn.Module=None, # o2p.decoder を仮定 デコーダはこのクラスの外で定義することにする
        #decoder:nn.Module=o2p.decoder,
        n_layers:int=1,                        # デコーダの中間層数
        #n_layers:int=config['n_layers'],      # デコーダの中間層数
        bidirectional:bool=False,              # デコーダを双方向モデルにするか否か
        #bidirectional:bool=config['bidirectional'], # デコーダを双方向モデルにするか否か
    )->None:

        super().__init__()

        # 外部で定義されたモデルを使う。外部で定義されたモデルへ影響が及ばないようにコピーして用いる
        self.decoder = copy.deepcopy(decoder)

        # 単語の意味ベクトル a.k.a 埋め込み表現 を decoder の中間層に接続するための変換層
        # 別解としては，入力層に接続する方法があるが，それはまた別実装にする
        self.enc_transform_layer = nn.Linear(
            in_features=inp_dim1+inp_dim2,
            out_features=n_hid)

        # デコーダ側の入力信号を埋め込み表現に変換
        self.decoder_emb = nn.Embedding(
            num_embeddings=dec_vocab_size,
            embedding_dim=n_hid,
            padding_idx=0)

        # デコーダ本体の定義
        self.decoder = nn.LSTM(
            input_size=n_hid,
            hidden_size=n_hid,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional)

        # 最終出力層
        self.bi_fact = 2 if bidirectional else 1
        self.out_layer = nn.Linear(self.bi_fact * n_hid, dec_vocab_size)

    def forward(self, vec1:torch.tensor, vec2:torch.tensor, dec_inp:torch.tensor):

        enc_inp = torch.cat((vec1, vec2),axis=1)
        #enc_inp = torch.cat((vec1, vec2),axis=1).to(device)
        enc_emb = self.enc_transform_layer(enc_inp)
        hnx, cnx = enc_emb.clone(), enc_emb.clone()
        hnx = hnx.unsqueeze(0)
        cnx = cnx.unsqueeze(0)

        if self.bi_fact == 2:
            hnx = hnx.repeat(2)
            cnx = cnx.repeat(2)

        dec_emb = self.decoder_emb(dec_inp)

        batch_size = enc_inp.size(0)
        exp_hid_size = self.decoder.get_expected_hidden_size(enc_inp, batch_sizes=[batch_size])
        dec_out, (hny, cny) = self.decoder(dec_emb,(hnx, cnx))

        return self.out_layer(dec_out)

# # 以下確認作業
# ds = os2p_ds
# os2p = VecVec2Seq(
#     inp_dim1=ds.w2v.vector_size, #+n_hid,
#     inp_dim2=ds.orth_vecs.shape[-1],
#     dec_vocab_size=len(ds.phoneme),
#     n_hid=config['n_hid'],
#     n_layers=config['n_layers'],
#     bidirectional=config['bidirectional']).to(device)
# print(os2p.eval())

def save_checkpoint(checkpoint_path, model):
    state = {'state_dict': model.state_dict() }
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print(f'model loaded from {checkpoint_path}')

import os
def fit_vecvec2seq_w_cpt(
    model:torch.nn.modules.module.Module=None,
    epochs:int=10,
    ds:Dataset=None,
    batch_size:int=256,
    #batch_size=config['batch_size'],
    collate_fn=_collate_fn,
    optimizer:torch.optim=None,
    criterion:torch.nn.modules.loss=nn.CrossEntropyLoss(ignore_index=-1),
    #interval:int=None,
    isPrint:bool=False,
    losses:list=None,
    isDraw:bool=True,
    device:str='cpu',
    #device:str=config['device'],
    lr:float=0.001,
    #lr:float=config['adam_lr'],
    betas:tuple=(0.9, 0.98),
    #betas:tuple=config['adam_betas'],
    eps:float=1e-09,
    #eps:float=config['adam_eps'],
    fname_saved_base = 'RAM/2023_1210'
    ):
    """ vecvec2seq の訓練に用いる関数"""

    start_time = time.time()   # 開始時刻の保存

    dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn)

    if losses == None:
        losses = []

    model.train()

    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

    for epoch in range(epochs):
        epoch_loss = 0.
        for _inp, _tch in dataloader:
            emb1, emb2, tch2 = [], [], []
            for (_emb1, _emb2), _tch in zip(_inp, _tch):
                emb1.append(_emb1.cpu().numpy())
                emb2.append(_emb2.cpu().numpy())
                tch2.append(_tch)
            emb1 = torch.tensor(np.array(emb1)).to(device)
            emb2 = torch.tensor(np.array(emb2)).to(device)
            tch2 = pad_sequence(tch2, batch_first=True).to(device)

            out = model(emb1, emb2, tch2).to(device)
            loss = criterion(out[0], tch2[0])
            epoch_loss += loss.item()
            for h in range(1,len(tch2)):
                loss += criterion(out[h], tch2[h])
                epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(epoch_loss/ds.__len__())
        print(f'epoch:{epoch:2d}',
              f'loss:{epoch_loss/ds.__len__():9.6f}')

        # save checkpoint
        #checkpoint_fname = os.path.join('RAM', '2023_1206os2p_epoch' + f'{epoch:02d}' + '.pt' )
        checkpoint_fname = fname_saved_base + f'{epoch:02d}' + '.pt'
        print(f'checkpoint_fname:{checkpoint_fname}')
        save_checkpoint(checkpoint_fname, model=model)


    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if isDraw:
        plt.plot(losses)
        plt.title(f'epochs:{epochs}, batch_size:{batch_size}, time collapsed:{total_time_str}')
        plt.show()

    return {'Training time':total_time_str,
            'losses': losses,
            'optimizer': optimizer,
            'time': total_time}

#res = fit_vecvec2seq_w_cpt(epochs=5, model=os2p, ds=os2p_ds)


def eval_vecvec2seq(
    model:torch.nn.modules.module.Module=None,
    ds:Dataset=None,
    isPrint:bool=False,
    errors:list=None,
    device:str='cpu'):

    model.eval()
    if errors == None:
        errors=[]

    corrects, incorrects = [], []
    #for N in range(ds.__len__()):
    for N in tqdm(range(ds.__len__())):

        (emb1, emb2), tch2 = ds.__getitem__(N)
        #(emb1, emb2), tch2 = os2p_ds.__getitem__(N)
        _o_ = model(
            emb1.unsqueeze(0).clone().detach().to(device),
            emb2.unsqueeze(0).clone().detach().to(device),
            tch2.unsqueeze(0).clone().detach().to(device))
        _tch = "".join(c for c in ds.phon_ids2phon(tch2.squeeze(0).detach().cpu().numpy()[1:-1]))
        _out = "".join(c for c in ds.phon_ids2phon(_o_.argmax(axis=2).squeeze(0).detach().cpu().numpy()[1:-1]))
        # _tch = "".join(c for c in os2p_ds.phon_ids2phon(tch2.squeeze(0).detach().cpu().numpy()[1:-1]))
        # _out = "".join(c for c in os2p_ds.phon_ids2phon(_o_.argmax(axis=2).squeeze(0).detach().cpu().numpy()[1:-1]))

        yesno = _tch == _out
        wrd = ds.getitem(N)[0]

        if yesno == True:
            corrects.append((N, wrd, _out, _tch))
        else:
            incorrects.append((N, wrd, _out, _tch))
            #_out = ds.target_ids2target(y_hat)
            errors.append((N, wrd, _out, _tch))
            if isPrint:
                color = 'grey' if isOK else 'red'
                wrd = ds.getitem(N)[0]
                print(colored(f'{N:05d}', color),
                      colored(wrd, color='grey'), # , attrs=["bold"]),
                      colored(_out, color, attrs=["bold"]),
                      colored(_tch, color, attrs=["bold"]))

    cr = len(corrects) / ds.__len__()
    return {'正解率': cr * 100,
            'エラー': incorrects,
            '正答': corrects}

# os2p_errors = eval_vecvec2seq(model=os2p, ds=os2p_ds)
# print(f"正解率:{os2p_errors['正解率']}")


def eval_vecvec2seq_wrds(
    model:torch.nn.modules.module.Module=None,  # 評価スべきモデル
    target_wrds:list=None,        # 評価すべき単語リスト
    ds:Dataset=None,              # 訓練したデータセット
    isPrint:bool=False,
    device:str='cpu',
    errors:list=None):

    model.eval()
    if errors == None:
        errors=[]

    corrects, incorrects, no_counts = [], [], []
    for wrd in target_wrds:

        if not wrd in ds.words:
            no_counts.append(wrd)
        else:
            idx = ds.words.index(wrd)
            #idx = os2p_ds.words.index(wrd)
            (emb1, emb2), tch2 = ds.__getitem__(idx)
            #(emb1, emb2), tch2 = os2p_ds.__getitem__(idx)
            _o_ = model(
                emb1.unsqueeze(0).clone().detach().to(device),
                emb2.unsqueeze(0).clone().detach().to(device),
                tch2.unsqueeze(0).clone().detach().to(device))
            _tch = "".join(c for c in ds.phon_ids2phon(tch2.squeeze(0).detach().cpu().numpy()[1:-1]))
            _out = "".join(c for c in ds.phon_ids2phon(_o_.argmax(axis=2).squeeze(0).detach().cpu().numpy()[1:-1]))
            #_tch = "".join(c for c in os2p_ds.phon_ids2phon(tch2.squeeze(0).detach().cpu().numpy()[1:-1]))
            #_out = "".join(c for c in os2p_ds.phon_ids2phon(_o_.argmax(axis=2).squeeze(0).detach().cpu().numpy()[1:-1]))

            yesno = _tch == _out
            color = 'grey' if yesno == True else 'red'

            if yesno == True:
                corrects.append((wrd,_out,_tch,yesno))
            else:
                incorrects.append((wrd,_out,_tch, yesno))
                errors.append((idx, wrd, _out, _tch))
                if isPrint:
                    color = 'grey' if isOK else 'red'
                    print(colored(wrd, color='grey'),
                          colored(_tch, color,attrs=["bold"]),
                          colored(_out, color, attrs=["bold"]),
                         )

    cr = len(corrects) / (len(corrects) + len(incorrects))
    return {'正解率': cr * 100,
            'no_counts': no_counts,
            'エラー': errors,
            '正解': corrects,
           }

# _errors = eval_vecvec2seq_wrds(model=os2p, ds=os2p_ds, target_wrds=kw_in)
# #_errors = eval_vecvec2seq_wrds(model=os2p, ds=os2p_ds, target_wrds=kw_all)
# #_errors = eval_vecvec2seq_wrds(model=os2p, ds=os2p_ds, target_wrds=RAM.fushimi1999_list[:])
# print(f"正解率:{_errors['正解率']}")
# print(len(kw_all))
# print(_errors)
