
# [Build your own Transformer from scratch using Pytorch](https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb)

# PyTorch を使用して基本的な Transformer モデルをゼロから構築する。
# Transformer モデルは Vaswani+ が論文 Attention is All You Need で導入したもので，機械翻訳やテキスト要約などの seq2seq 課題のために設計された深層学習アーキテクチャである。
# 自己注意機構に基づいており，GPT や BERT など，多くの最先端の自然言語処理モデルの基盤となっている。
# <!-- In this tutorial, we will build a basic Transformer model from scratch using PyTorch.
# The Transformer model, introduced by Vaswani et al. in the paper “Attention is All You Need,” is a deep learning architecture designed for sequence-to-sequence tasks, such as machine translation and text summarization.
# It is based on self-attention mechanisms and has become the foundation for many state-of-the-art natural language processing models, like GPT and BERT. -->

# Transformer の詳細は，以下の 2 記事を参照:
# <!-- To understand Transformer models in detail kindly visit these two articles: -->

# 1. [All you need to know about ‘Attention’ and ‘Transformers’ — In-depth Understanding — Part 1](https://medium.com/towards-data-science/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-1-552f0b41d021)
# 2. [All you need to know about ‘Attention’ and ‘Transformers’ — In-depth Understanding — Part 2](https://medium.com/towards-data-science/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-2-bf2403804ada)

# Transformer モデルの作成においては以下の段階を踏む:<!-- To build our Transformer model, we’ll follow these steps:-->

# 1.  必要なライブラリやモジュールをインポートする
# 2.  基本的な構成要素を定義する: マルチヘッド注意，位置ごとのフィードフォワードネットワーク，および，位置符号化器
# 3. 符号化器と復号化器の層を構築
# 4. 符号化器と復号化器の層を組み合わせて，完全な transformer モデルを作成する。
# 5. サンプルデータの作成
# 6. モデルの訓練

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

# マルチヘッド注意
class MultiHeadAttention(nn.Module):
	"""入力パラメータと線形変換層でモジュールを初期化する。
	注意得点を計算し，入力テンソルを複数のヘッドに再整形し，すべてのヘッドからの注意出力を結合する。
	`forward` メソッドはマルチヘッド自己注意を計算し，モデルが入力系列の別の面に注意を向けることを可能にする。"""
	def __init__(self,
				 model_dim:int,	# 各層の素子数
				 num_heads:int   # ヘッド数，マルチヘッド注意の定義に必要
				):
		super().__init__()
		assert model_dim % num_heads == 0, "model_dim は num_heads で割り切れる数である必要がある"

		self.model_dim = model_dim
		self.num_heads = num_heads
		self.d_k = model_dim // num_heads

		self.W_q = nn.Linear(in_features=model_dim, out_features=model_dim)
		self.W_k = nn.Linear(in_features=model_dim, out_features=model_dim)
		self.W_v = nn.Linear(in_features=model_dim, out_features=model_dim)
		self.W_o = nn.Linear(in_features=model_dim, out_features=model_dim)

	def scaled_dot_product_attention(self,
									 Q:torch.Tensor,
									 K:torch.Tensor,
									 V:torch.Tensor,
									 mask=None):
		attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
		if mask is not None:
			attn_scores = attn_scores.masked_fill(mask==0, -1e9)
		attn_probs = torch.softmax(attn_scores, dim=-1)
		output = torch.matmul(attn_probs, V)
		return output

	def split_heads(self,
					x:torch.Tensor):
		batch_size, seq_length, model_dim = x.size()
		return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

	def combine_heads(self,
					  x:torch.Tensor):
		batch_size, _, seq_length, d_k = x.size()
		return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.model_dim)

	def forward(self,
				Q:torch.Tensor,
				K:torch.Tensor,
				V:torch.Tensor, mask=None):
		Q = self.split_heads(self.W_q(Q))
		K = self.split_heads(self.W_k(K))
		V = self.split_heads(self.W_v(V))

		attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
		output = self.W_o(self.combine_heads(attn_output))
		return output


# 位置ごとのフィードフォワードネットワーク Position-wise_FeedForward_Networks
class PositionWiseFeedForward(nn.Module):
	"""`PositionWiseFeedForward` クラスは、PyTorchの `nn.Module` を拡張し，位置ごとのフィードフォワードネットワークの実装である。
	このクラスは，2 つの線形変換層と ReLU 活性化関数で初期化される。
	`forward` メソッドは，これらの変換と活性化関数を順次適用して出力を計算する。
	この処理により，モデルは入力要素の位置を考慮しながら予測を行うことができる。"""
	def __init__(self,
				 model_dim:int,
				 ff_dim:int):
		super().__init__()
		self.fc1 = nn.Linear(in_features=model_dim, out_features=ff_dim)
		self.fc2 = nn.Linear(in_features=ff_dim, out_features=model_dim)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x
		#return self.fc2(self.relu(self.fc1(x)))


# 位置符号化器 Positional Encoding
class PositionalEncoding(nn.Module):
	"""位置符号化器 Positional Encoding
	位置符号化器は，入力系列の各トークンの位置情報を挿入するために使用される。
	異なる周波数の正弦波関数と余弦波関数を使用して位置情報を生成する。
	符号化器層は，マルチヘッド注意層，位置ごとのフィードフォワード層，2 つの層正規化層で構成される。
	`PositionalEncoding` クラスは，入力パラメータ `model_dim` と `max_seq_length` で初期化し，位置符号化器の値を格納するテンソルを作成する。
	このクラスは，スケール因子 `div_term` に基づいて，偶数インデックスと奇数インデックスの正弦波と余弦波の値をそれぞれ計算する。
	forward メソッドは，格納された位置符号化値を入力テンソルに追加することで位置符号化を計算し，モデルが入力配列の位置情報を捕捉できるようにする。"""
	def __init__(self,
				 model_dim:int,
				 max_seq_length:int):
		super().__init__()

		pe = torch.zeros(max_seq_length, model_dim)
		position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		self.register_buffer('pe', pe.unsqueeze(0))

	def forward(self, x):
		return x + self.pe[:, :x.size(1)]


# 符号化器層 Encoder Layer
class EncoderLayer(nn.Module):
	"""EncoderLayer クラスは，入力パラメータと，`MultiHeadAttention` モジュール，`PositionWiseFeedForward` モジュール，2 つの層正規化モジュール，ドロップアウト層などの成分で初期化する。
	forward メソッドは，自己注意を適用して符号化層の出力を計算し，注意出力を入力テンソルに加え，その結果を正規化する。
	次に，位置ごとのフィードフォワード出力を計算し，正規化された自己注意出力と結合し，最終結果を正規化してから処理されたテンソルを返す。"""
	def __init__(self, model_dim, num_heads, ff_dim, dropout):

		super().__init__()
		self.self_attn = MultiHeadAttention(model_dim, num_heads)
		self.feed_forward = PositionWiseFeedForward(model_dim, ff_dim)
		self.norm1 = nn.LayerNorm(model_dim)
		self.norm2 = nn.LayerNorm(model_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, mask):
		attn_output = self.self_attn(x, x, x, mask)
		x = self.norm1(x + self.dropout(attn_output))
		ff_output = self.feed_forward(x)
		x = self.norm2(x + self.dropout(ff_output))
		return x


## 復号化器層 Decoder Layer
# 復号化器層は，2 つのマルチヘッド注意層，位置ごとのフィードフォワード層，3 つの層正規化層で構成される。
class DecoderLayer(nn.Module):
	"""復号化器層は，入力パラメータと，マスクされた自己注意と交差注意のためのマルチヘッド注意モジュール，位置ごとのフィードフォワードモジュール，3  層の正規化モジュール，およびドロップアウト層などの成分で初期化する。
	forward メソッドは，以下のステップを実行することで，復号化器層の出力を計算する:

	1. マスクされた自己注意出力を計算し，入力テンソルに加算した後，ドロップアウトと層正規化を行う。
	2. 復号化器出力と符号化器出力の間の交差注意出力を計算し，正規化されたマスクされた自己注意出力に加え，ドロップアウトと層正規化を行う。
	3. 位置ごとのフィードフォワード出力を計算し，正規化された交差注意出力に加え，ドロップアウトと層正規化を行う。
	4. 処理されたテンソルを返す。"""
	def __init__(self, model_dim, num_heads, ff_dim, dropout):
		super().__init__()

		self.self_attn = MultiHeadAttention(model_dim, num_heads)
		self.cross_attn = MultiHeadAttention(model_dim, num_heads)
		self.feed_forward = PositionWiseFeedForward(model_dim, ff_dim)
		self.norm1 = nn.LayerNorm(model_dim)
		self.norm2 = nn.LayerNorm(model_dim)
		self.norm3 = nn.LayerNorm(model_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, enc_output, src_mask, tgt_mask):
		attn_output = self.self_attn(x, x, x, tgt_mask)
		x = self.norm1(x + self.dropout(attn_output))
		attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
		x = self.norm2(x + self.dropout(attn_output))
		ff_output = self.feed_forward(x)
		x = self.norm3(x + self.dropout(ff_output))
		return x

# Transformer Model
# 上記により，復号化は入力と符号化出力に基づいて標的系列を生成することができる。
class Transformer(nn.Module):
	"""Transformer クラスは，先に定義されたモジュールを組み合わせて，完全な Transformer モデルを作成する。
	初期化の際，Transformer モジュールは入力パラメータを設定し，ソースとターゲット系列用の埋め込み層，PositionalEncoding モジュール，スタック層を作成する EncoderLayer とDecoderLayer モジュール，復号化器出力を映し出すための線形層，ドロップアウト層など様々な成分を初期化する。

	generate_mask メソッドは，パディングトークンを無視し，復号化器が将来のトークンに注目しないように，ソースとターゲット系列に二値化マスクを作成する。
	forward メソッドは，以下のステップで Transformer モデルの出力を計算する：

	1. generate_maskメソッドでソースマスクとターゲットマスクを生成する。
	2. ソースとターゲットの埋め込みを計算し，位置符号化とドロップアウトを適用する。
	3. ソース系列を符号化層で処理し，enc_output テンソルを更新する。
	4. 符号化出力とマスクを用いて，ターゲット系列を復号化器層で処理し，dec_output テンソルを更新する。
	5. 復号化器出力に線形射影層を適用し，出力ロジットを得る。"""
	def __init__(self,
				 src_vocab_size:int,
				 tgt_vocab_size:int,
				 model_dim:int,
				 num_heads:int,
				 num_layers:int,
				 ff_dim:int,
				 max_seq_length:int,
				 dropout:float):
		super().__init__()
		self.encoder_embedding = nn.Embedding(src_vocab_size, model_dim)
		self.decoder_embedding = nn.Embedding(tgt_vocab_size, model_dim)
		self.positional_encoding = PositionalEncoding(model_dim, max_seq_length)

		self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
		self.decoder_layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

		self.fc = nn.Linear(model_dim, tgt_vocab_size)
		self.dropout = nn.Dropout(dropout)

	def generate_mask(self, src, tgt):
		src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
		tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
		seq_length = tgt.size(1)
		nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
		tgt_mask = tgt_mask & nopeak_mask
		return src_mask, tgt_mask

	def forward(self, src, tgt):
		src_mask, tgt_mask = self.generate_mask(src, tgt)
		src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
		tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

		enc_output = src_embedded
		for enc_layer in self.encoder_layers:
			enc_output = enc_layer(enc_output, src_mask)

		dec_output = tgt_embedded
		for dec_layer in self.decoder_layers:
			dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

		output = self.fc(dec_output)
		return output



