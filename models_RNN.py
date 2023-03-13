import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

device="cuda:0" if torch.cuda.is_available() else "cpu"

class EncoderLSTM(nn.Module):
    """LSTM による符号化器
    EncoderRNN をベースにして，LSTM に拡張
    """
    def __init__(self, vocab_size:int, n_hid:int):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_hid = n_hid
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_hid)
        # GRU, LSTM, SRN を抽象化して RNN と命名してみる
        self.rnn = nn.LSTM(input_size=n_hid, hidden_size=n_hid, batch_first=True)

    def forward(self,
                inp:torch.Tensor,
                hid:Tuple[torch.Tensor,torch.Tensor],
                device=device):
        # 原作は，1 トークンづつだから，.view(1,1,-1) なのだろう
        embs = self.emb_layer(inp).view(1, 1, -1)
        out = embs
        out, hid = self.rnn(out, hid)
        return out, hid

    def initHidden(self)->torch.Tensor:
        hid = torch.zeros(1, 1, self.n_hid)
        cel = torch.zeros(1, 1, self.n_hid)
        return hid, cel


class AttnDecoderLSTM(nn.Module):
    """注意付き復号化器の定義"""
    def __init__(self, n_hid:int, vocab_size:int, max_length:int, dropout_p:float=0.0):
        super().__init__()

        self.n_hid = n_hid
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_hid)

        # n_hid + n_hid -> max_length の線形層を定義
        self.attn = nn.Linear(in_features=n_hid * 2, out_features=max_length)

        # n_hid + n_hid -> n_hid の線形層
        self.attn_combine = nn.Linear(in_features=n_hid * 2, out_features=n_hid)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.LSTM(input_size=n_hid, hidden_size=n_hid, batch_first=True)
        self.out_layer = nn.Linear(in_features=n_hid, out_features=vocab_size)

    def forward(self,
                inp:torch.Tensor,
                hid:Tuple[torch.Tensor, torch.Tensor],
                encoder_outputs:torch.Tensor,
                device=device):
        embs = self.emb_layer(inp).view(1, 1, -1)
        embs = self.dropout(embs)

        # 注意の重みを計算
        # 入力 (embedded[0]) と 中間層 (hidden[0]) とを第 2 次元に沿って連結 (torch.cat(dim=1))
        #print(f' embs[0].size():{embs[0].size()}\n',
        #      f'hid[0][0].size():{hid[0][0].size()}')
        attn_weights = F.softmax(self.attn(torch.cat((embs[0], hid[0][0]), 1)), dim=1)
        #attn_weights = F.softmax(self.attn(torch.cat((embs[0], hid[0]), 1)), dim=1)

        # エンコーダの中間層の値を計算:= 注意の重み (attn_weights) * エンコーダの出力 (hiden) = 注意を描けた
        # この attn_weights.unsqueeze(0) は 第 1 次元が batch になっているようだ
        #print(f'attn_weights.unsqueeze(0).size():{attn_weights.unsqueeze(0).size()}',
        #      f'encoder_outputs.unsqueeze(0).size():{encoder_outputs.unsqueeze(0).size()}')
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        out = torch.cat((embs[0], attn_applied[0]), 1)
        out = self.attn_combine(out).unsqueeze(0)
        out = F.relu(out)
        out, hid = self.rnn(out, hid)
        out = F.log_softmax(self.out_layer(out[0]), dim=1)
        return out, hid, attn_weights

    def initHidden(self)->torch.Tensor:
        hid = torch.zeros(1, 1, self.n_hid)
        cel = torch.zeros(1, 1, self.n_hid)
        return hid, cell

class DecoderLSTM(nn.Module):
    """注意付き復号化器の定義"""
    def __init__(self, n_hid:int, vocab_size:int, max_length:int, dropout_p:float=0.0):
        super().__init__()

        self.n_hid = n_hid
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_hid)

        # n_hid + n_hid -> max_length の線形層を定義
        self.attn = nn.Linear(in_features=n_hid * 2, out_features=max_length)

        # n_hid + n_hid -> n_hid の線形層
        self.attn_combine = nn.Linear(in_features=n_hid * 2, out_features=n_hid)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.LSTM(input_size=n_hid, hidden_size=n_hid, batch_first=True)
        self.out_layer = nn.Linear(in_features=n_hid, out_features=vocab_size)

    def forward(self,
                inp:torch.Tensor,
                hid:Tuple[torch.Tensor, torch.Tensor],
                encoder_outputs:torch.Tensor,
                device=device):
        embs = self.emb_layer(inp).view(1, 1, -1)
        #embs = self.dropout(embs)
        attn_weights = F.softmax(self.attn(torch.cat((embs[0], hid[0][0]), 1)), dim=1)
        #attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        #out = torch.cat((embs[0], attn_applied[0]), 1)
        #out = self.attn_combine(out).unsqueeze(0)
        #out = F.relu(out)
        out = embs
        out, hid = self.rnn(out, hid)
        out = F.log_softmax(self.out_layer(out[0]), dim=1)
        return out, hid, attn_weights

    def initHidden(self)->torch.Tensor:
        hid = torch.zeros(1, 1, self.n_hid)
        cel = torch.zeros(1, 1, self.n_hid)
        return hid, cell


class EncoderSRN(nn.Module):
    """SRN による符号化器
    EncoderRNN をベースにして SRN に変更
    """
    def __init__(self, vocab_size:int, n_hid:int):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_hid = n_hid
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_hid)
        # GRU, LSTM, SRN を抽象化して RNN と命名してみる
        self.rnn = nn.RNN(input_size=n_hid, hidden_size=n_hid, batch_first=True)

    def forward(self,
                inp:torch.Tensor,
                hid:torch.Tensor,
                device=device):
        embs = self.emb_layer(inp).view(1, 1, -1)
        out = embs
        out, hid = self.rnn(out, hid)
        return out, hid

    def initHidden(self)->torch.Tensor:
        hid = torch.zeros(1, 1, self.n_hid)
        return hid


class AttnDecoderSRN(nn.Module):
    """注意付き復号化器の定義 SRN 版"""
    def __init__(self, n_hid:int, vocab_size:int, max_length:int, dropout_p:float=0.0):
        super().__init__()

        self.n_hid = n_hid
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_hid)

        # n_hid + n_hid -> max_length の線形層を定義
        self.attn = nn.Linear(in_features=n_hid * 2, out_features=max_length)

        # n_hid + n_hid -> n_hid の線形層
        self.attn_combine = nn.Linear(in_features=n_hid * 2, out_features=n_hid)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.RNN(input_size=n_hid, hidden_size=n_hid, batch_first=True)
        self.out_layer = nn.Linear(in_features=n_hid, out_features=vocab_size)

    def forward(self,
                inp:torch.Tensor,
                hid:torch.Tensor,
                encoder_outputs:torch.Tensor,
                device=device):
        embs = self.emb_layer(inp).view(1, 1, -1)
        embs= self.dropout(embs)

        # 注意の重みを計算
        # 入力 (embedded[0]) と 中間層 (hidden[0]) とを第 2 次元に沿って連結 (torch.cat(dim=1))
        #print(f' embs[0].size():{embs[0].size()}\n',
        #      f'hid[0][0].size():{hid[0][0].size()}')
        #attn_weights = F.softmax(self.attn(torch.cat((embs[0], hid[0][0]), 1)), dim=1)
        attn_weights = F.softmax(self.attn(torch.cat((embs[0], hid[0]), 1)), dim=1)

        # エンコーダの中間層の値を計算:= 注意の重み (attn_weights) * エンコーダの出力 (hiden) = 注意を描けた
        # この attn_weights.unsqueeze(0) は 第 1 次元が batch になっているようだ
        #print(f'attn_weights.unsqueeze(0).size():{attn_weights.unsqueeze(0).size()}',
        #      f'encoder_outputs.unsqueeze(0).size():{encoder_outputs.unsqueeze(0).size()}')
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        out = torch.cat((embs[0], attn_applied[0]), 1)
        out = self.attn_combine(out).unsqueeze(0)
        out = F.relu(out)
        out, hid = self.rnn(out, hid)
        out = F.log_softmax(self.out_layer(out[0]), dim=1)
        return out, hid, attn_weights

    def initHidden(self)->torch.Tensor:
        hid = torch.zeros(1, 1, self.n_hid)
        return hid

class DecoderSRN(nn.Module):
    """注意付き復号化器の定義 SRN 版"""
    def __init__(self, n_hid:int, vocab_size:int, max_length:int, dropout_p:float=0.0):
        super().__init__()

        self.n_hid = n_hid
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_hid)

        # n_hid + n_hid -> max_length の線形層を定義
        self.attn = nn.Linear(in_features=n_hid * 2, out_features=max_length)

        # n_hid + n_hid -> n_hid の線形層
        self.attn_combine = nn.Linear(in_features=n_hid * 2, out_features=n_hid)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.RNN(input_size=n_hid, hidden_size=n_hid, batch_first=True)
        self.out_layer = nn.Linear(in_features=n_hid, out_features=vocab_size)

    def forward(self,
                inp:torch.Tensor,
                hid:torch.Tensor,
                encoder_outputs:torch.Tensor,
                device=device):
        embs = self.emb_layer(inp).view(1, 1, -1)
        attn_weights = F.softmax(self.attn(torch.cat((embs[0], hid[0]), 1)), dim=1)
        #attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        #out = torch.cat((embs[0], attn_applied[0]), 1)
        #out = self.attn_combine(out).unsqueeze(0)
        #out = F.relu(out)
        out = embs
        out, hid = self.rnn(out, hid)
        out = F.log_softmax(self.out_layer(out[0]), dim=1)
        return out, hid, attn_weights

    def initHidden(self)->torch.Tensor:
        hid = torch.zeros(1, 1, self.n_hid)
        return hid

class EncoderGRU(nn.Module):
    """GRU による符号化器"""
    def __init__(self, vocab_size:int, n_hid:int):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_hid = n_hid
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_hid)
        # GRU, LSTM, SRN を抽象化して RNN と命名してみる
        self.rnn = nn.GRU(input_size=n_hid, hidden_size=n_hid, batch_first=True)

    def forward(self,
                inp:torch.Tensor,
                hid:Tuple[torch.Tensor,torch.Tensor],
                device=device):
        # 原作は，1 トークンづつだから，.view(1,1,-1) なのだろう
        embs = self.emb_layer(inp).view(1, 1, -1)
        out = embs
        out, hid = self.rnn(out, hid)
        return out, hid

    def initHidden(self)->torch.Tensor:
        return torch.zeros(1, 1, self.n_hid, device=device)


class AttnDecoderGRU(nn.Module):
    """注意付き復号化器の定義"""
    """注意付き復号化器の定義"""
    def __init__(self, n_hid:int, vocab_size:int, max_length:int, dropout_p:float=0.0):
        super().__init__()

        self.n_hid = n_hid
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_hid)

        # n_hid + n_hid -> max_length の線形層を定義
        self.attn = nn.Linear(in_features=n_hid * 2, out_features=max_length)

        # n_hid + n_hid -> n_hid の線形層
        self.attn_combine = nn.Linear(in_features=n_hid * 2, out_features=n_hid)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.GRU(input_size=n_hid, hidden_size=n_hid, batch_first=True)
        self.out_layer = nn.Linear(in_features=n_hid, out_features=vocab_size)


    def forward(self,
                inp:torch.Tensor,
                hid:torch.Tensor,
                encoder_outputs:torch.Tensor,
                device=device):
        embs = self.emb_layer(inp).view(1, 1, -1)
        embs= self.dropout(embs)

        attn_weights = F.softmax(self.attn(torch.cat((embs[0], hid[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        out = torch.cat((embs[0], attn_applied[0]), 1)
        out = self.attn_combine(out).unsqueeze(0)
        out = F.relu(out)
        out, hid = self.rnn(out, hid)
        out = F.log_softmax(self.out_layer(out[0]), dim=1)
        return out, hid, attn_weights

    def initHidden(self)->torch.Tensor:
        return torch.zeros(1, 1, self.n_hid, device=device)


class DecoderGRU(nn.Module):
    """注意付き復号化器の定義"""
    """注意付き復号化器の定義"""
    def __init__(self, n_hid:int, vocab_size:int, max_length:int, dropout_p:float=0.0):
        super().__init__()

        self.n_hid = n_hid
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_hid)

        # n_hid + n_hid -> max_length の線形層を定義
        self.attn = nn.Linear(in_features=n_hid * 2, out_features=max_length)

        # n_hid + n_hid -> n_hid の線形層
        self.attn_combine = nn.Linear(in_features=n_hid * 2, out_features=n_hid)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.GRU(input_size=n_hid, hidden_size=n_hid, batch_first=True)
        self.out_layer = nn.Linear(in_features=n_hid, out_features=vocab_size)


    def forward(self,
                inp:torch.Tensor,
                hid:torch.Tensor,
                encoder_outputs:torch.Tensor,
                device=device):
        embs = self.emb_layer(inp).view(1, 1, -1)
        #embs = self.dropout(embs)
        attn_weights = F.softmax(self.attn(torch.cat((embs[0], hid[0]), 1)), dim=1)
        #attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        #out = torch.cat((embs[0], attn_applied[0]), 1)
        #out = self.attn_combine(out).unsqueeze(0)
        #out = F.relu(out)
        out = embs
        out, hid = self.rnn(out, hid)
        out = F.log_softmax(self.out_layer(out[0]), dim=1)
        return out, hid, attn_weights

    def initHidden(self)->torch.Tensor:
        return torch.zeros(1, 1, self.n_hid, device=device)


"""
encoderLSTM = EncoderLSTM(vocab_size=len(ds.source_list), n_hid=configs['hidden_size'])
decoderLSTM = AttnDecoderLSTM(
    vocab_size=len(ds.target_list),
    n_hid=configs['hidden_size'],
    dropout_p=configs['dropout_p'],
    max_length=ds.maxlen).to(device)

encoderSRN = EncoderSRN(vocab_size=len(ds.source_list), n_hid=configs['hidden_size'])
decoderSRN = AttnDecoderSRN(
    vocab_size=len(ds.target_list),
    n_hid=configs['hidden_size'],
    dropout_p=configs['dropout_p'],
    max_length=ds.maxlen).to(device)

from RAM import EncoderRNN, AttnDecoderRNN
encoderGRU = EncoderRNN(n_inp=len(ds.source_list), n_hid=configs['hidden_size'])
decoderGRU = AttnDecoderRNN(
    n_out=len(ds.target_list),
    n_hid=configs['hidden_size'],
    dropout_p=configs['dropout_p'],
    max_length=ds.maxlen).to(device)

encoderGRU_optimizer=configs['optim_func'](encoderGRU.parameters(), lr=configs['lr'])
decoderGRU_optimizer=configs['optim_func'](decoderGRU.parameters(), lr=configs['lr'])
encoderSRN_optimizer=configs['optim_func'](encoderSRN.parameters(), lr=configs['lr'])
decoderSRN_optimizer=configs['optim_func'](decoderSRN.parameters(), lr=configs['lr'])
encoderLSTM_optimizer=torch.optim.Adam(encoderLSTM.parameters(), lr=configs['lr'])
decoderLSTM_optimizer=torch.optim.Adam(decoderLSTM.parameters(), lr=configs['lr'])
"""

# encoder0 = EncoderRNN(
#     n_inp=len(ds.source_list),
#     n_hid=configs['hidden_size'])

# decoder0 = AttnDecoderRNN(
#     n_out=len(ds.target_list),
#     n_hid=configs['hidden_size'],
#     dropout_p=configs['dropout_p'],
#     max_length=ds.maxlen).to(device)
# encoder_optimizer=torch.optim.Adam(encoder.parameters(), lr=configs['lr'])
# decoder_optimizer=torch.optim.Adam(decoder.parameters(), lr=configs['lr'])
