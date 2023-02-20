import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.nn.functional as F

from termcolor import colored
import math
import random
import numpy as np
import time
import math
import jaconv
import sys

import operator
import matplotlib.pyplot as plt
import japanize_matplotlib

device="cuda:0" if torch.cuda.is_available() else "cpu"

def draw_word_char_histgram(
    _dict:dict=None, #psylex71.train_data,
    key:str='orig',  # or 'phon'
    title:str=None,
    topN:int=100,
    figsize=(20,4),
    figsize2=(14,4)):

    if title == None:
        title = key
    chr_count, len_count = {}, {}
    for k, v in _dict.items():
        wrd = v[key]
        wrd_len = len(wrd)
        for ch in wrd:
            if ch in chr_count:
                chr_count[ch] += 1
            else:
                chr_count[ch] = 1

        if wrd_len in len_count:
            len_count[wrd_len] += 1
        else:
            len_count[wrd_len] = 1

    N_chr=np.array([v for v in chr_count.values()]).sum()

    if topN > len(chr_count):
        topN = len(chr_count)

    chr_count_sorted = sorted(chr_count.items(), key=operator.itemgetter(1), reverse=True)
    plt.figure(figsize=figsize)
    plt.bar(range(topN), [x[1]/N_chr for x in chr_count_sorted[:topN]])
    plt.xticks(ticks=range(topN), labels=[c[0] for c in chr_count_sorted[:topN]])

    if topN == len(chr_count):
        plt.title(f'{title}項目頻度')
    else:
        plt.title(f'{title}項目頻度 (上位:{topN} 語)')
    plt.ylabel('相対頻度')
    plt.show()

    N_len=np.array([v for v in len_count.values()]).sum()
    len_count_sorted = sorted(len_count.items(), key=operator.itemgetter(0), reverse=False)
    plt.figure(figsize=figsize2)
    plt.bar(range(len(len_count_sorted)), [x[1]/N_len for x in len_count_sorted])
    plt.xticks(ticks=range(len(len_count_sorted)), labels=[c[0] for c in len_count_sorted])
    plt.ylabel(f'{title}相対頻度')
    plt.title(f'{title}項目長頻度')
    plt.show()


def eval_input_seq2seq(
    inp_wrd:str='////',
    encoder:torch.nn.Module=None,    decoder:torch.nn.Module=None,
    ds:torch.utils.data.Dataset=None, topN:int=2, isPrint:bool=True):

    if inp_wrd == '////':
        if ds.source == 'orth':
            inp_wrd = input('単語を入力:')
        elif ds.source == 'phon':
            inp_wrd = input('｢よみ｣をひらがなで入力:')
            inp_wrd = jaconv.hiragana2julius(inp_wrd).split()

    _input_ids = ds.source_tkn2ids(inp_wrd) + [ds.source_list.index('<EOW>')]
    _output_words, _output_ids, _attentions = evaluate(
        encoder=encoder, decoder=decoder,
        input_ids=_input_ids,
        max_length=ds.maxlen,
        source_vocab=ds.source_list,
        target_vocab=ds.target_list,
    )
    dec_wrds = _output_words

    if isPrint:
        print(f'出力:{dec_wrds}') # np.array(dec_vals)}')
        #print(f'出力:{dec_wrds}, likelihood:{likelihood}') # np.array(dec_vals)}')
    return dec_wrds, None
    #return dec_wrds, likelihood


def convert_ids2tensor(
    sentence_ids:list,
    device:torch.device=device):

    """数値 ID リストをテンソルに変換
    例えば，[0,1,2] -> tensor([[0],[1],[2]])
    """
    return torch.tensor(sentence_ids, dtype=torch.long, device=device).view(-1, 1)


def asMinutes(s:int)->str:
    """時間変数を見やすいように，分と秒に変換して返す"""
    m = math.floor(s / 60)
    s -= m * 60
    return f'{int(m):2d}分 {int(s):2d}秒'


def timeSince(since:time.time,
            percent:time.time)->str:
    """開始時刻 since と，現在の処理が全処理中に示す割合 percent を与えて，経過時間と残り時間を計算して表示する"""
    now = time.time()  #現在時刻を取得
    s = now - since    # 開始時刻から現在までの経過時間を計算
    #s = since - now
    es = s / (percent) # 経過時間を現在までの処理割合で割って終了予想時間を計算
    rs = es - s        # 終了予想時刻から経過した時間を引いて残り時間を計算

    return f'経過時間:{asMinutes(s)} (残り時間 {asMinutes(rs)})'


def calc_accuracy(
    encoder, decoder,
    _dataset,
    max_length=None,
    source_vocab=None, target_vocab=None,
    source_ids=None, target_ids=None,
    isPrint=False):

    ok_count = 0
    for i in range(_dataset.__len__()):
        _input_ids, _target_ids = _dataset.__getitem__(i)
        _output_words, _output_ids, _attentions = evaluate(
            encoder=encoder,
            decoder=decoder,
            input_ids=_input_ids,
            max_length=max_length,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            source_ids=source_ids,
            target_ids=target_ids,
        )
        ok_count += 1 if _target_ids == _output_ids else 0
        if (_target_ids != _output_ids) and (isPrint):
            print(i, _target_ids == _output_ids, _output_words, _input_ids, _target_ids)

    return ok_count/_dataset.__len__()


def evaluate(
    encoder:torch.nn.Module, decoder:torch.nn.Module,
    input_ids:list=None,
    source_vocab:list=None, target_vocab:list=None,
    max_length:int=1, device:torch.device=device):

    with torch.no_grad():
        input_tensor = convert_ids2tensor(input_ids)
        input_length = input_tensor.size()[0]
        encoder_hid = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.n_hid, device=device)

        for ei in range(input_length):
            encoder_out, encoder_hid = encoder(input_tensor[ei], encoder_hid)
            encoder_outputs[ei] += encoder_out[0, 0]

        decoder_inp = torch.tensor([[source_vocab.index('<SOW>')]], device=device)
        decoder_hid = encoder_hid

        decoded_words, decoded_ids = [], []  # decoded_ids を追加
        decoder_attns = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_out, decoder_hid, decoder_attn = decoder(
                decoder_inp, decoder_hid, encoder_outputs, device=device)
            decoder_attns[di] = decoder_attn.data
            topv, topi = decoder_out.data.topk(1)
            decoded_ids.append(int(topi.squeeze().detach())) # decoded_ids に追加
            if topi.item() == target_vocab.index('<EOW>'):
                decoded_words.append('<EOW>')
                break
            else:
                decoded_words.append(target_vocab[topi.item()])

            decoder_inp = topi.squeeze().detach()

        return decoded_words, decoded_ids, decoder_attns[:di + 1]  # decoded_ids を返すように変更


def check_a_dataset(
    encoder:torch.nn.Module=None,
    decoder:torch.nn.Module=None,
    _dataset:torch.utils.data.Dataset=None,
    max_length:int=0,
    source_vocab:list=None,
    target_vocab:list=None,
    device:torch.device=device):

    if (_dataset == None) or (encoder == None) or (decoder == None) or (max_length == 0) or (source_vocab == None):
        return

    ret = []
    ok_count = 0
    for i in range(_dataset.__len__()):
        _input_ids, _target_ids = _dataset.__getitem__(i)
        _output_words, _output_ids, _attentions = evaluate(
            encoder=encoder, decoder=decoder,
            source_vocab=source_vocab, target_vocab=target_vocab,
            input_ids=_input_ids, max_length=max_length, device=device)
        ok_count += 1 if _target_ids == _output_ids else 0
    ret.append(f'{ok_count/_dataset.__len__():.3f}')
    return ret


def check_vals_performance(
    encoder:torch.nn.Module=None,
    decoder:torch.nn.Module=None,
    _dataset:torch.utils.data.Dataset=None,
    max_length:int=0,
    source_vocab:list=None,
    target_vocab:list=None,
    device:torch.device=device):

    if (_dataset == None) or (encoder == None) or (decoder == None) or (max_length == 0) or (source_vocab == None):
        return

    ret = []
    for _x in _dataset:
        ok_count = 0
        for i in range(_dataset[_x].__len__()):
            #_input_ids, _target_ids = _dataset.__getitem__(i)
            _input_ids, _target_ids = _dataset[_x].__getitem__(i)
            _output_words, _output_ids, _attentions = evaluate(
                encoder=encoder, decoder=decoder,
                source_vocab=source_vocab, target_vocab=target_vocab,
                input_ids=_input_ids,
                max_length=max_length,
                #source_ids=source_ids, #target_ids=target_ids,
                device=device)
            ok_count += 1 if _target_ids == _output_ids else 0
        #print(f'{_x}:{ok_count/_dataset.__len__():.3f},',end="")
        #print(f'{_x}:{ok_count/_dataset[_x].__len__():.3f},',end=" ")
        ret.append(f'{_x}:{ok_count/_dataset[_x].__len__():.3f}')
    #print()
    return ret


# シミュレーションに必要なパラメータの設定用辞書のプロトタイプ
params = {
    #'dataset_name'  : 'vdrj',   # ['pyslex71', 'vdrj', 'onechar', 'fushimi1999']
    'dataset_name'  : 'fushimi1999',   # ['pyslex71', 'vdrj', 'onechar', 'fushimi1999']
    #'dataset_name'   : 'onechar',
    'traindata_size':  1000,    # 訓練データ (語彙) 数，
    #'traindata_size':  20000,   # 訓練データ (語彙) 数，
    'traindata_ratio': 0.9,     # 訓練データと検証データを分割する比率。ただし onechar データセットでは無効

    'epochs': 30,               # 学習のためのエポック数

    # 以下 `source` と `rget` を定義することで，別の課題を実行可能
    'source': 'phon',          # ['orth', 'phon']
    'target': 'phon',          # ['orth', 'phon']

    #'hidden_size': 256,        # 中間層のニューロン数
    #'hidden_size': 128,
    'hidden_size': 32,

    'random_seed': 42,          # 乱数の種。ダグラス・アダムス著「銀河ヒッチハイカーズガイド」

    'pretrained': False,       # True であれば訓練済ファイルを読み込む
    #'isTrain'   : True,       # True であれば学習する

    'verbose'   : False,
    # 学習済のモデルパラメータを保存するファイル名
    #'path_saved': '2022_0607lam_o2p_hid32_vocab10k.pt',
    'path_saved': '2023_0201RAM.pt',
    #'path_saved': False,                      # 保存しない場合

    'lr': 1e-4,                       # 学習率
    #'lr': 1e-3,                       # 学習率
    'dropout_p': 0.0,                 # ドロップアウト率
    'teacher_forcing_ratio': 0.5,     # 教師強制を行う確率
    'optim_func': torch.optim.Adam,   # 最適化アルゴリズム ['torch.optim.Adam', 'torch.optim.SGD', 'torch.optim.AdamW']
    'loss_func' :torch.nn.NLLLoss(),  # 負の対数尤度損失 ['torch.nn.NLLLoss()', or 'torch.nn.CrossEntropyLoss()']
}

from .dataset import *

def dup_model_with_params(params:dict=params,
                          verbose:bool=False):
    # データセットの読み込み
    # 1. Psylex71 は「NTT 日本語の語彙特性」頻度表であり，著作権上の問題があるため配布不可
    # 2. VDRJ は松下言語学習ラボ，[日本語を読むための語彙データベース（研究用）](http://www17408ui.sakura.ne.jp/tatsum/database.html#vdrj) を加工して作成したデータである
    # 3. OneChar は一文字の読みについてのおもちゃのデータセットである。
    #
    # RAM ディレクトリ直下に，`psylex71_data.gz`, `vdrj_data.gz` がある。
    # これらは，`RAM/make_psylex71_dict.py`, `RAM/make_vdrj_dict.py` を実行して作成されたデータファイルである。
    # ここでは，これらのデータファイルが作成済と仮定している。
    if params['dataset_name'] == 'psylex71':
        psylex71_dataset = Psylex71_Dataset(source=params['source'], target=params['target'], max_words=params['traindata_size'])
        ds = psylex71_dataset
    elif params['dataset_name'] == 'vdrj':
        vdrj_dataset     = VDRJ_Dataset(source=params['source'], target=params['target'], max_words=params['traindata_size'])
        ds = vdrj_dataset
    elif params['dataset_name'] == 'onechar':
        onechar_dataset  = OneChar_Dataset(source=params['source'], target=params['target'])
        ds = onechar_dataset
    elif params['dataset_name'] == 'fushimi1999':
        fushimi1999_dataset = Fushimi1999_Dataset(source=params['source'], target=params['target'])
        ds = fushimi1999_dataset
    else:
        psylex71_dataset = Psylex71_Dataset(source=params['source'], target=params['target'], max_words=params['traindata_size'])
        ds = psylex71_dataset


    # 符号化器-復号化器モデルの定義
    encoder = EncoderRNN(
        n_inp=len(ds.source_list),               # 符号化器への入力データ次元数の特徴数 (語彙数): int
        n_hid=params['hidden_size']).to(device)  # 符号化器の中間層数，埋め込みベクトルとして復号化器へ渡される次元数: int
                                                 # 復号化器の出力層素子数は，入力層と同一であるので指定しない

    decoder = AttnDecoderRNN(
        n_hid=params['hidden_size'],             # 復号化器の中間層次元数: int
        n_out=len(ds.target_list),               # 復号化器の出力層次元数，入力層の次元と等しいので入力層次元を指定せず: int
        dropout_p=params['dropout_p'],
        max_length=ds.maxlen).to(device)

    ## 訓練用最適化関数の定義
    encoder_optimizer = params['optim_func'](encoder.parameters(), lr=params['lr'])
    decoder_optimizer = params['optim_func'](decoder.parameters(), lr=params['lr'])

    ## データを訓練データと検証データとに分割
    N_train = int(ds.__len__() * params['traindata_ratio'])   # 訓練データを 90 % に相当する数に
    N_val   = ds.__len__() - N_train    # 検証データを残り 10 % に相当する数
    if params['dataset_name'] != 'onechar':
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset=ds,
            lengths=(N_train, N_val),
            generator=torch.Generator().manual_seed(params['random_seed']))
    else:
        train_dataset = ds
        val_dataset = None
        N_train = len(ds.data_dict)
        N_val = 0

    color = 'grey'
    for k, v in params.items():
        print(f'{k}:{colored(v, color=color, attrs=["bold"])}')

    print('train_dataset size:', colored(f'{N_train}', color, attrs=['bold']),
          'val_dataset size:', colored(f'{N_val}', color, attrs=['bold'])) if verbose else None

    return {'data_dict':ds.data_dict,
            'encoder':encoder,
            'decoder':decoder,
            'encoder_optimizer': encoder_optimizer,
            'decoder_optimizer': decoder_optimizer,
            'train_dataset':train_dataset,
            'val_dataset': val_dataset,
            'params': params}


# X = dup_model_with_params(params=params, verbose=False)
# ds2, encoder2, decoder2 = X['dataset'], X['encoder'], X['decoder'],
# encoder2_optimizer, decoder2_optimizer = X['encoder_optimizer'], X['decoder_optimizer']
# train_dataset2, val_dataset2 = X['train_dataset'], X['val_dataset']
# ds2.__len__()
def check_fushimi1999_words(encoder:torch.nn.Module,
                            decoder:torch.nn.Module,
                            ds:torch.utils.data.Dataset,
                            fushimi1999_list:list=fushimi1999_list[:120],
                            cr_every:int=4):
    ok_count = 0
    for i, wrd in enumerate(fushimi1999_list):
        _end = ", " if ((i+1) % cr_every) != 0 else '\n'

        if wrd in ds.orth2info_dict:
            _inp = ds.orth2info_dict[wrd][ds.source]
            _out = eval_input_seq2seq(ds=ds, encoder=encoder, decoder=decoder, inp_wrd=_inp, isPrint=False)
            _out_wrd = _out[0][:-1]
            _tch = ds.orth2info_dict[wrd][ds.target]
            is_ok = _tch == _out_wrd
            if is_ok:
                ok_count += 1

                _out_wrd = "".join(c for c in _out_wrd)
                _tch_wrd = ''.join(c for c in _tch)

                color = "blue" if is_ok else "red"
                print(f'{i+1:03d} {wrd}',
                      f":/{_out_wrd}/({_tch_wrd})",
                      colored(f'{is_ok}', color, attrs=["bold"]), end=_end)
            else:
                print(f'{i+1:03d} {wrd}', f'/{"".join(c for c in _out_wrd)}/', '-' * 12, end=_end)
        else:
            print(f'{i+1:03d}', colored(f'{wrd} 訓練データに存在しない', color='cyan', attrs=['bold']), end=_end)

    N = i + 1
    color='green'
    print('正解率:',colored(f'{ok_count/N * 100:.3f}%=({ok_count}/{N})', color, attrs=['bold']))


from .dataset import *
# 符号化器-復号化器モデルの定義
from .model import EncoderRNN
from .model import AttnDecoderRNN

def set_model_from_param_file(fname:str='2023_0213ram.pt'):
    if os.path.exists(fname):
        X = torch.load(fname)
    else:
        print(f'{fname} does not exists')
        sys.exit()

    params = X['params']
    # データセットの読み込み
    # 1. Psylex71 は「NTT 日本語の語彙特性」頻度表であり，著作権上の問題があるため配布不可
    # 2. VDRJ は [日本語を読むための語彙データベース（研究用）](http://www17408ui.sakura.ne.jp/tatsum/database.html#vdrj) を加工して作成したデータ
    # 3. OneChar は一文字の読みについてのおもちゃのデータセット
    # RAM ディレクトリ直下に，`psylex71_data.gz`, `vdrj_data.gz` がある。
    # これらは，`RAM/make_psylex71_dict.py`, `RAM/make_vdrj_dict.py` を実行して作成されたデータファイルである。
    # ここでは，これらのデータファイルが作成済と仮定している。
    if params['dataset_name'] == 'psylex71':
        psylex71_dataset = Psylex71_Dataset(source=params['source'],
                                            target=params['target'],
                                            max_words=params['traindata_size'])
        ds = psylex71_dataset
    elif params['dataset_name'] == 'vdrj':
        vdrj_dataset     = VDRJ_Dataset(source=params['source'],
                                        target=params['target'],
                                        max_words=params['traindata_size'])
        ds = vdrj_dataset
    elif params['dataset_name'] == 'onechar':
        onechar_dataset  = OneChar_Dataset(source=params['source'],
                                           target=params['target'])
        ds = onechar_dataset
    elif params['dataset_name'] == 'fushimi1999':
        fushimi1999_dataset = Fushimi1999_Dataset(source=params['source'],
                                                  target=params['target'])
        ds = fushimi1999_dataset
    else:
        psylex71_dataset = Psylex71_Dataset(source=params['source'], target=params['target'],
                                            max_words=params['traindata_size'])
        ds = psylex71_dataset

    encoder = EncoderRNN(
        n_inp=len(ds.source_list),               # 符号化器への入力データ次元数の特徴数 (語彙数): int
        n_hid=params['hidden_size']).to(device)  # 符号化器の中間層数，埋め込みベクトルとして復号化器へ渡される次元数: int
                                                 # 復号化器の出力層素子数は，入力層と同一であるので指定しない

    decoder = AttnDecoderRNN(
        n_hid=params['hidden_size'],    # 復号化器の中間層次元数: int
        n_out=len(ds.target_list),      # 復号化器の出力層次元数，入力層の次元と等しいので入力層次元を指定せず: int
        dropout_p=params['dropout_p'],
        max_length=ds.maxlen).to(device)

    ## 訓練用最適化関数の定義
    encoder_optimizer = params['optim_func'](encoder.parameters(), lr=params['lr'])
    decoder_optimizer = params['optim_func'](decoder.parameters(), lr=params['lr'])

    encoder.load_state_dict(X['encoder'])
    decoder.load_state_dict(X['decoder'])
    encoder_optimizer.load_state_dict(X['encoder_optimizer'])
    decoder_optimizer.load_state_dict(X['decoder_optimizer'])

    return encoder, decoder, encoder_optimizer, decoder_optimizer, params, ds


# from RAM import train_one_seq2seq
# from RAM import train_epochs

import os

def set_params_from_config(
    json_fname:str="",
    configs:str=None,
    device:torch.device=device):

    X = set_params_from_file(
        json_fname=json_fname,
        params=configs,
        device=device)

    ret = {
        'params': X[0],
        'encoder': X[1],
        'decoder': X[2],
        'dataset': X[3],
        'train_dataset': X[4],
        'val_dataset': X[5],
        'encoder_optimizer': X[6],
        'decoder_optimizer': X[7],
        'N_train': X[8],
        'N_val': X[9]}
    return ret

def set_params_from_file(json_fname:str="",
                         params:dict=None,
                         device:torch.device=device):

    if json_fname != "":
        try:
            os.path.exists(json_fname)
        except:
            print(f"file {json_fname} does not exist")
            sys.exit()

        with open(json_fname, 'r') as fp:
            params = json.load(fp)
    elif params == None:
        print('params is None')
        sys.exit()

    stop_list = [] if params.get('stop_list')==None else params['stop_list']
    # データセットの読み込み
    # 1. Psylex71 は「NTT 日本語の語彙特性」頻度表であり，著作権上の問題があるため配布不可
    # 2. VDRJ は松下言語学習ラボ，[日本語を読むための語彙データベース（研究用）](http://www17408ui.sakura.ne.jp/tatsum/database.html#vdrj) を加工して作成したデータである
    # 3. OneChar は一文字の読みについてのおもちゃのデータセットである。
    #
    # RAM ディレクトリ直下に，`psylex71_data.gz`, `vdrj_data.gz` がある。
    # これらは，`RAM/make_psylex71_dict.py`, `RAM/make_vdrj_dict.py` を実行して作成されたデータファイルである。
    # ここでは，これらのデータファイルが作成済と仮定している。
    if params['dataset_name'] == 'psylex71':
        psylex71_dataset = Psylex71_Dataset(source=params['source'], target=params['target'],
                                            max_words=params['traindata_size'],
                                            stop_list=stop_list)
                                            #stop_list=fushimi1999_list[:120])
                                            # fushimi1999_list[:120] としているのは 240 以降の単語は非単語である。
                                            # このため，検証データとしては不適なため
        ds = psylex71_dataset
    elif params['dataset_name'] == 'vdrj':
        vdrj_dataset     = VDRJ_Dataset(source=params['source'], target=params['target'],
                                        max_words=params['traindata_size'],
                                        stop_list=stop_list)
                                        #stop_list=fushimi1999_list[:120])
        ds = vdrj_dataset
    elif params['dataset_name'] == 'onechar':
        onechar_dataset  = OneChar_Dataset(source=params['source'], target=params['target'])
        ds = onechar_dataset
    elif params['dataset_name'] == 'fushimi1999':
        fushimi1999_dataset = Fushimi1999_Dataset(source=params['source'], target=params['target'])
        ds = fushimi1999_dataset
    else:
        psylex71_dataset = Psylex71_Dataset(source=params['source'], target=params['target'],
                                            max_words=params['traindata_size'],
                                            stop_list=stop_list)
                                            #stop_list=fushimi1999_list[:120])
        ds = psylex71_dataset


    encoder = EncoderRNN(
        n_inp=len(ds.source_list),               # 符号化器への入力データ次元数の特徴数 (語彙数): int
        n_hid=params['hidden_size']).to(device)  # 符号化器の中間層数，埋め込みベクトルとして復号化器へ渡される次元数: int
                                                 # 復号化器の出力層素子数は，入力層と同一であるので指定しない

    decoder = AttnDecoderRNN(
        n_hid=params['hidden_size'],             # 復号化器の中間層次元数: int
        n_out=len(ds.target_list),               # 復号化器の出力層次元数，入力層の次元と等しいので入力層次元を指定せず: int
        dropout_p=params['dropout_p'],
        max_length=ds.maxlen).to(device)

    ## データを訓練データと検証データとに分割
    N_train = int(ds.__len__() * params['traindata_ratio'])   # 訓練データを 90 % に相当する数に
    N_val   = ds.__len__() - N_train    # 検証データを残り 10 % に相当する数
    if (params['dataset_name'] == 'onechar') or (params['dataset_name'] == 'fushimi1999'):
        train_dataset = ds
        val_dataset = None
        N_train = len(ds.data_dict)
        N_val = 0
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset=ds,
            lengths=(N_train, N_val),
            generator=torch.Generator().manual_seed(params['random_seed']))

    ## 訓練用最適化関数の定義
    if isinstance(params['optim_func'], str):
        params['optim_func'] = eval(params['optim_func'])

    encoder_optimizer = params['optim_func'](encoder.parameters(), lr=params['lr'])
    decoder_optimizer = params['optim_func'](decoder.parameters(), lr=params['lr'])
    params['loss_func'] = eval(params['loss_func'])()
    #params['loss_func'] = str(params['loss_func'])

    return params, encoder, decoder, ds, train_dataset, val_dataset, \
            encoder_optimizer, decoder_optimizer, N_train, N_val


import datetime

def save_model_and_params(
    params:dict=params,
    models:dict=None,
    isColab:bool=isColab,
    force:bool=True,
    device="cuda:0" if torch.cuda.is_available() else "cpu"):

    fname = params['path_saved']
    if isColab:
        # colab 上では，Gdrive 上に保存
        fname = 'drive/MyDrive/' + fname

    if os.path.exists(fname) and (force!=True):
        print(f'{fname} というファイルが存在し，かつ，force オプションが {force} であるため保存しません')
        return


    encoder, decoder = models['encoder'], models['decoder']
    encoder_optimizer, decoder_optimizer = models['encoder_optimizer'], models['decoder_optimizer']
    losses = models['losses']

    timestamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    torch.save({'encoder':encoder.state_dict(),
                'decoder':decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'params':params,
                'losses':losses,
                'timestamp': timestamp}, fname)

    ret = torch.load(fname, map_location=torch.device(device))
    return ret


def save_model_and_configs(
    configs:dict=params,
    force:bool=True):

    fname = configs['path_saved']
    isColab = configs.get('isColab', False)
    if isColab:
        # colab 上では，Gdrive 上に保存
        fname = 'drive/MyDrive/' + fname

    if os.path.exists(fname) and (force!=True):
        print(f'{fname} というファイルが存在し，かつ，force オプションが {force} であるため保存しません')
        return

    encoder, decoder = configs['encoder'], configs['decoder']
    encoder_optimizer, decoder_optimizer = configs['encoder_optimizer'], configs['decoder_optimizer']
    losses = models['losses']

    timestamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    torch.save({'encoder':encoder.state_dict(),
                'decoder':decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'configs':configs,
                'losses':losses,
                'timestamp': timestamp}, fname)

    ret = torch.load(fname, map_location=torch.device(device))
    return ret
