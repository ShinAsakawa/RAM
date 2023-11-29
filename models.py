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
    device:str="cpu",
    ):
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
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    if interval == None:
        interval = int(ds.__len__()/batch_size) >> 2

    for epoch in range(epochs):
        i = 0
        for _inp, _tch in dataloader:
            enc_inp = pad_sequence(_inp, batch_first=True).to(device)
            dec_inp = pad_sequence(_tch, batch_first=True).to(device)
            tch = pad_sequence(_tch, batch_first=True, padding_value=-1.0).to(device)
            out = model(enc_inp, dec_inp)
            loss = criterion(out[0], tch[0])
            for h in range(1,len(tch)):
                loss += criterion(out[h], tch[h])
            losses.append(loss.item()/len(_inp))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            if (i % interval) == 0:
                print(f'epoch:{epoch+1:2d}',
                      f'batch:{i:5d}',
                      f'loss:{loss.item()/batch_size:.5f}')

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

# fit_seq2seq(epochs=1, model=o2p, ds=psylex71_ds_o2p); # 音読モデル
# fit_seq2seq(epochs=1, model=p2p, ds=psylex71_ds_p2p); # 復唱モデル
# fit_seq2seq(epochs=1, model=p2o, ds=psylex71_ds_p2o); # ディクテーションモデル
# fit_seq2seq(epochs=1, model=o2o, ds=psylex71_ds_o2o); # 写字モデル


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
            errors.append((N, wrd, _out,y_hat))
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
    #model:torch.nn.modules.module.Module=o2p,
    epochs:int=10,
    ds:Dataset=None,
    #ds:Dataset=psylex71_ds_o2s,
    batch_size=10,
    #batch_size=batch_size,
    collate_fn=_collate_fn,
    #dataloader:torch.utils.data.dataloader.DataLoader=dl_o2s,
    optimizer:torch.optim=None,
    lr:float=0.001,
    criterion_dec:torch.nn.modules.loss=nn.MSELoss(),
    criterion_enc:torch.nn.modules.loss=nn.CrossEntropyLoss(ignore_index=-1),
    interval:int=None,
    isPrint:bool=False,
    losses:list=None,
    isDraw:bool=True,
    device:str="cpu",
    ):
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
        out_vec, _ = model(enc_inp).to('cpu')
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
