import torch
import jaconv
import gzip
import json
import os
import sys
import re
import MeCab

from IPython import get_ipython
isColab =  'google.colab' in str(get_ipython())
if isColab:
    mecab_yomi = MeCab.Tagger('-Oyomi').parse
else:
	mecab_dic_dirs = {
	    'Pasiphae': '/usr/local/lib/mecab/dic/mecab-ipadic-neologd',
	    'Sinope':' /opt/homebrew/lib/mecab/dic/ipadic',
	    'Leda': '/usr/local/lib/mecab/dic/ipadic',
	    'colab': '/usr/share/mecab/dic/ipadic'
	}


hostname = 'colab' if isColab else os.uname().nodename.split('.')[0]
if hostname == 'colab':
    wakati = MeCab.Tagger('-Owakati').parse
    mecab_yomi = MeCab.Tagger('-Oyomi').parse
else:
    mecab_dic_dir = mecab_dic_dirs[hostname]
    wakati = MeCab.Tagger(f'-Owakati -d {mecab_dic_dir}').parse
    mecab_yomi = MeCab.Tagger(f'-Oyomi -d {mecab_dic_dir}').parse


from .fushimi1999 import _fushimi1999_list
fushimi1999_list = _fushimi1999_list()


class RAM_Dataset(torch.utils.data.Dataset):
    """
    Simulation のための Pytorch Dataset クラスの super class の定義
    アルファベット，カタカナ，ひらがな，常用漢字 (または区点コードによる JIS 第一水準) などを共通で定義したいために，
    ちなみに，常用漢字は，以前は 1945 文字であった。現在は，2136 字である。
    詳細は https://ja.wikipedia.org/wiki/%E5%B8%B8%E7%94%A8%E6%BC%A2%E5%AD%97%E4%B8%80%E8%A6%A7 などを参照
    """

    def __init__(self,
                 source:str="orth",
                 target:str="phon",
                 mecab_yomi=mecab_yomi,
                 max_words:int=200000,
                ):
        super().__init__()

        self.jchar_list = self.get_jchar_list()
        self.phon_list = self.get_phon_list()
        self.joyo_list = self.get_joyo_chars()
        self.source=source
        self.target=target
        self.set_source_and_target_from_param(source=source, target=target)
        self.yomi=mecab_yomi

        self.orth_list = self.joyo_list
        self.max_words = max_words

        c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）
        c5 = '[フ][ュ]'
        mora_cond = '('+c5+'|'+c1+'|'+ c2 + '|' + c3 + '|' + c4 +')'
        self.re_mora = re.compile(mora_cond)


    def kana2moraWakachi(self, kana:str):
        """https://qiita.com/shimajiroxyz/items/a133d990df2bc3affc12"""
        return self.re_mora.findall(kana)


    def is_phon_vocab(self, _phn:list)->bool:
        for _p in _phn:
            if not _p in self.phon_list:
                return False
        return True


    def kata2phon(self, kata:str)->list:
        if kata == 'ュウ':
            kata = 'ユウ'

        _hira = jaconv.kata2hira(kata)
        _juli = jaconv.hiragana2julius(_hira).split()

        if self.is_phon_vocab(_juli):
            return _juli
        else:
            _mora_wakachi = self.kana2moraWakachi(kata)
            _juli_new = []
            for ch in _mora_wakachi:
                if ch in self.phon_list:
                    _juli_new.append(ch)
                else:
                    hira_ch_yomi = jaconv.kata2hira(self.yomi(ch).strip())
                    _juli_new += jaconv.hiragana2julius(hira_ch_yomi).split()
            return _juli_new


    def __len__(self, **kwargs):
        """ Must be overrided """
        raise NotImplementedError

    def __getitem(self, **kwargs):
        """ Must be overrided """
        raise NotImplementedError

    def set_source_and_target_from_params(
        self,
        source:str='orth',
        target:str='phon',
        is_print:bool=True):
        # ソースとターゲットを設定

        if source == 'orth':
            self.source_list = self.orth_list
            self.source_maxlen = self.orth_maxlen
            self.source_ids2tkn = self.orth_ids2tkn
            self.source_tkn2ids = self.orth_tkn2ids
        elif source == 'phon':
            self.source_list = self.phon_list
            self.source_maxlen = self.phon_maxlen
            self.source_ids2tkn = self.phon_ids2tkn
            self.source_tkn2ids = self.phon_tkn2ids

        if target == 'orth':
            self.target_list = self.orth_list
            self.target_maxlen = self.orth_maxlen
            self.target_ids2tkn = self.orth_ids2tkn
            self.target_tkn2ids = self.orth_tkn2ids
        elif target == 'phon':
            self.target_list = self.phon_list
            self.target_maxlen = self.phon_maxlen
            self.target_ids2tkn = self.phon_ids2tkn
            self.target_tkn2ids = self.phon_tkn2ids


    def orth2orth_ids(self, orth:str):
        orth_ids = [self.orth_list.index(ch) if ch in self.orth_list else self.orth_list.index('<UNK>') for ch in orth]
        return orth_ids

    def phon2phon_ids(self, phon:list):
        phon_ids = [self.phon_list.index(ph) if ph in self.phon_list else self.phon_list.index('<UNK>') for ph in phon]
        return phon_ids

    def orth_ids2tkn(self, ids: list):
        return [self.orth_list[idx] for idx in ids]

    def orth_tkn2ids(self, tkn: list):
        return [self.orth_list.index(_tkn) if _tkn in self.orth_list else self.orth_list.index('<UNK>') for _tkn in tkn]

    def phon_ids2tkn(self, ids: list):
        return [self.phon_list[idx] for idx in ids]

    def phon_tkn2ids(self, tkn: list):
        return [self.phon_list.index(_tkn) if _tkn in self.phon_list else self.phon_list.index('<UNK>') for _tkn in tkn]


    def get_alphabet_upper_chars(self):
        return 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'

    def get_alphabet_lower_chars(self):
        return 'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'

    def get_num_chars(self):
        return '０１２３４５６７８９'

    def get_hira_chars(self):
        return 'ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをん'

    def get_kata_chars(self):
        return 'ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶ'

    def get_kuten_chars(self):
        from .char_ja import kuten as kuten
        return kuten().chars

    def get_joyo_chars(self):
        from .char_ja import chars_joyo as chars_joyo
        joyo_chars = chars_joyo().char_list
        return joyo_chars


    def get_jchar_list(self):

        _kata=self.get_kata_chars()
        _hira=self.get_hira_chars()
        _num =self.get_num_chars()
        _alpha_upper=self.get_alphabet_upper_chars()
        _alpha_lower=self.get_alphabet_lower_chars()
        _joyo_chars =self.get_joyo_chars()
        jchar_list = ['<PAD>', '<EOW>', '<SOW>', '<UNK>']
        for x in [_num, _alpha_upper, _alpha_lower, _kata, _hira, _joyo_chars, 'ー〇ゞ々']:
            for ch in x:
                jchar_list.append(ch)
        return jchar_list


    def get_phon_list(self):
        return ['<PAD>', '<EOW>', '<SOW>', '<UNK>',
               'N', 'a', 'a:', 'e', 'e:', 'i', 'i:', 'i::', 'o', 'o:', 'o::', 'u', 'u:',
               'b', 'by', 'ch', 'd', 'dy', 'f', 'g', 'gy', 'h', 'hy', 'j', 'k', 'ky',
               'm', 'my', 'n', 'ny', 'p', 'py', 'q', 'r', 'ry', 's', 'sh', 't', 'ts', 'w', 'y', 'z'] + ['ty', ':']



class Psylex71_Dataset(RAM_Dataset):

    def __init__(self,
                 source:str="orth",
                 target:str="phon",
                 max_words:int=20000,
                 stop_list:list=fushimi1999_list[:120],
                ):

        self.datasetname = 'psylex71'
        psylex71_data_fname = 'RAM/psylex71_data.gz'
        vdrj_data_fname = 'RAM/vdrj_data.gz'
        with gzip.open(psylex71_data_fname, 'rb') as zipfile:
            _X = json.loads(zipfile.read().decode('utf-8'))
        psylex71_dict = _X['dict']
        psylex71_freq = _X['freq']

        self.max_words = max_words
        self.phon_list = super().get_phon_list()
        self.jchar_list = super().get_jchar_list()
        self.source = source
        self.target = target
        self.orth_list = self.jchar_list

        data_dict, orth_maxlen, phon_maxlen = {}, 0, 0
        for wrd in psylex71_freq:
            if not wrd in stop_list:
                idx = len(data_dict)
                val = psylex71_dict[wrd]
                data_dict[idx] = val

                orth_len = len(psylex71_dict[wrd]['orth'])
                if orth_len > orth_maxlen:
                    orth_maxlen = orth_len
                phon_len = len(psylex71_dict[wrd]["phon"])
                if phon_len > phon_maxlen:
                    phon_maxlen = phon_len

                if idx >= (max_words -1):
                    break

        self.orth_maxlen = orth_maxlen + 1
        self.phon_maxlen = phon_maxlen + 1

        if self.orth_maxlen > self.phon_maxlen:
            self.maxlen = orth_maxlen
        else:
            self.maxlen = self.phon_maxlen

        self.data_dict = data_dict
        self.orth2info_dict = psylex71_dict
        super().set_source_and_target_from_params(source=source, target=target)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, x:int, **kwargs):
        _inp = self.data_dict[x][self.source]
        _tch = self.data_dict[x][self.target]
        _inp_ids = self.source_tkn2ids(_inp)
        _tch_ids = self.target_tkn2ids(_tch)
        return _inp_ids + [self.source_list.index('<EOW>')], _tch_ids + [self.target_list.index('<EOW>')]


class VDRJ_Dataset(RAM_Dataset):

    def __init__(self,
                 source:str="orth",
                 target:str="phon",
                 max_words:int=20000,
                 cover_rate:float=0.99,
                 stop_list:list=fushimi1999_list[:120]):


        self.datasetname = 'vdrj'

        vdrj_data_fname = 'RAM/vdrj_data.gz'
        with gzip.open(vdrj_data_fname, 'rb') as zipfile:
            _X = json.loads(zipfile.read().decode('utf-8'))
        vdrj_dict = _X['dict']

        self.max_words = max_words
        self.phon_list = super().get_phon_list()
        self.jchar_list = super().get_jchar_list()
        self.joyo_chars = super().get_joyo_chars()
        self.source = source
        self.target = target
        self.orth_list = self.jchar_list
        self.cover_rate=cover_rate

        # data_dict の key はインデックス番号とする
        data_dict, orth_maxlen, phon_maxlen = {}, 0, 0
        for k, v in vdrj_dict.items():
            wrd = v['lexeme']
            if not wrd in stop_list:
                idx = len(data_dict)
                data_dict[idx] = v

                orth_len = len(v['lexeme'])
                if orth_len > orth_maxlen:
                    orth_maxlen = orth_len
                phon_len = len(v['phon'])
                if phon_len > phon_maxlen:
                    phon_maxlen = phon_len
                if idx >= (max_words -1):
                    break
                if v['cover_r'] > cover_rate:
                    break

        self.orth_maxlen = orth_maxlen + 1
        self.phon_maxlen = phon_maxlen + 1
        if self.orth_maxlen > self.phon_maxlen:
            self.maxlen = orth_maxlen
        else:
            self.maxlen = self.phon_maxlen

        self.data_dict = data_dict
        orth2info_dict = {}
        for k, v in vdrj_dict.items():
            orth = v['lexeme']
            orth2info_dict[orth] = v
        self.orth2info_dict = orth2info_dict
        super().set_source_and_target_from_params(source=source, target=target)


    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, x:int, **kwargs):
        _inp = self.data_dict[x][self.source]
        _tch = self.data_dict[x][self.target]
        _inp_ids = self.source_tkn2ids(_inp)
        _tch_ids = self.target_tkn2ids(_tch)
        return _inp_ids + [self.source_list.index('<EOW>')], _tch_ids + [self.target_list.index('<EOW>')]


class OneChar_Dataset(RAM_Dataset):

    def __init__(self,
                 source:str="orth",
                 target:str="phon",
                 max_words:int=20000,
                 mecab_yomi=mecab_yomi,
                 stop_list:list=None):

        self.datasetname = 'vdrj'
        self.source = source
        self.target = target
        _num =super().get_num_chars()
        _alpha_upper=super().get_alphabet_upper_chars()
        _alpha_lower=super().get_alphabet_lower_chars()
        _hira=super().get_hira_chars()
        jchar_list = ['<PAD>', '<EOW>', '<SOW>', '<UNK>']
        for x in [_num, _alpha_upper, _hira]:
            for ch in x:
                jchar_list.append(ch)

        self.jchar_list = jchar_list
        self.max_words = len(jchar_list)
        self.phon_list = super().get_phon_list()
        self.orth_list = self.jchar_list

        c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）
        c5 = '[フ][ュ]'
        mora_cond = '('+c5+'|'+c1+'|'+ c2 + '|' + c3 + '|' + c4 +')'
        self.re_mora = re.compile(mora_cond)
        self.mecab_yomi = mecab_yomi

        # data_dict の key はインデックス番号とする
        data_dict, orth_maxlen, phon_maxlen = {}, 0, 0
        for wrd in jchar_list[4:]:
            _kata = mecab_yomi(wrd).strip()
            _phon = self.kata2phon(_kata)
            _idx = len(data_dict)
            phon_len = len(_phon)
            if phon_len > phon_maxlen:
                phon_maxlen = phon_len

            data_dict[_idx] = {
                'orth': wrd,
                'phon': _phon,
            }

        self.orth_maxlen = 1 + 1
        self.phon_maxlen = phon_maxlen + 1

        if self.orth_maxlen > self.phon_maxlen:
            self.maxlen = orth_maxlen
        else:
            self.maxlen = self.phon_maxlen

        self.data_dict = data_dict

        orth2info_dict = {}
        for k, v in data_dict.items():
            orth = v['orth']
            orth2info_dict[orth] = v
        self.orth2info_dict = orth2info_dict
        super().set_source_and_target_from_params(source=source, target=target)


    def orth2phon(self, _orth):
        _kata = mecab_yomi(_orth).strip()
        _phon = self.kata2phon(_kata)
        return _phon



    def kana2moraWakachi(self, kana:str):
        """https://qiita.com/shimajiroxyz/items/a133d990df2bc3affc12"""
        return self.re_mora.findall(kana)


    def is_phon_vocab(self, _phn:list)->bool:
        for _p in _phn:
            if not _p in self.phon_list:
                return False
        return True


    def kata2phon(self, kata:str)->list:
        if kata == 'ュウ':
            kata = 'ユウ'

        _hira = jaconv.kata2hira(kata)
        _juli = jaconv.hiragana2julius(_hira).split()

        if self.is_phon_vocab(_juli):
            return _juli
        else:
            _mora_wakachi = self.kana2moraWakachi(kata)
            _juli_new = []
            for ch in _mora_wakachi:
                if ch in self.phon_list:
                    _juli_new.append(ch)
                else:
                    hira_ch_yomi = jaconv.kata2hira(self.yomi(ch).strip())
                    _juli_new += jaconv.hiragana2julius(hira_ch_yomi).split()
            return _juli_new



    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, x:int, **kwargs):
        _inp = self.data_dict[x][self.source]
        _tch = self.data_dict[x][self.target]
        _inp_ids = self.source_tkn2ids(_inp)
        _tch_ids = self.target_tkn2ids(_tch)
        return _inp_ids + [self.source_list.index('<EOW>')], _tch_ids + [self.target_list.index('<EOW>')]



class Fushimi1999_Dataset(RAM_Dataset):
    """Fushimi, Ijuin, and Patternson (1999) の検査語彙のうち実在語 120 語だけを取り出したデータセット"""

    def __init__(self,
                 source:str="orth",
                 target:str="phon",
                 max_words:int=120,
                 stop_list:list=None):

        self.datasetname = 'fushimi1999'

        self.max_words = max_words
        self.phon_list = super().get_phon_list()
        self.jchar_list = super().get_jchar_list()
        self.joyo_chars = super().get_joyo_chars()
        self.source = source
        self.target = target
        self.orth_list = self.jchar_list

        # data_dict の key はインデックス番号とする
        data_dict={0: {"orth": "戦争", "ヨミ": "センソウ", "phon": ["s", "e", "N", "s", "o:"]},
                   1: {"orth": "倉庫", "ヨミ": "ソウコ", "phon": ["s", "o:", "k", "o"]},
                   2: {"orth": "医学", "ヨミ": "イガク", "phon": ["i", "g", "a", "k", "u"]},
                   3: {"orth": "注意", "ヨミ": "チュウイ", "phon": ["ch", "u:", "i"]},
                   4: {"orth": "記念", "ヨミ": "キネン", "phon": ["k", "i", "n", "e", "N"]},
                   5: {"orth": "番号", "ヨミ": "バンゴウ", "phon": ["b", "a", "N", "g", "o:"]},
                   6: {"orth": "料理", "ヨミ": "リョウリ", "phon": ["ry", "o:", "r", "i"]},
                   7: {"orth": "完全", "ヨミ": "カンゼン", "phon": ["k", "a", "N", "z", "e", "N"]},
                   8: {"orth": "開始", "ヨミ": "カイシ", "phon": ["k", "a", "i", "sh", "i"]},
                   9: {"orth": "印刷", "ヨミ": "インサツ", "phon": ["i", "N", "s", "a", "ts", "u"]},
                   10: {"orth": "連続", "ヨミ": "レンゾク", "phon": ["r", "e", "N", "z", "o", "k", "u"]},
                   11: {"orth": "予約", "ヨミ": "ヨヤク", "phon": ["y", "o", "y", "a", "k", "u"]},
                   12: {"orth": "多少", "ヨミ": "タショウ", "phon": ["t", "a", "sh", "o:"]},
                   13: {"orth": "教員", "ヨミ": "キョウイン", "phon": ["ky", "o:", "i", "N"]},
                   14: {"orth": "当局", "ヨミ": "トウキョク", "phon": ["t", "o:", "ky", "o", "k", "u"]},
                   15: {"orth": "材料", "ヨミ": "ザイリョウ", "phon": ["z", "a", "i", "ry", "o:"]},
                   16: {"orth": "夕刊", "ヨミ": "ユウカン", "phon": ["y", "u:", "k", "a", "N"]},
                   17: {"orth": "労働", "ヨミ": "ロウドウ", "phon": ["r", "o:", "d", "o:"]},
                   18: {"orth": "運送", "ヨミ": "ウンソウ", "phon": ["u", "N", "s", "o:"]},
                   19: {"orth": "電池", "ヨミ": "デンチ", "phon": ["d", "e", "N", "ch", "i"]},
                   20: {"orth": "反対", "ヨミ": "ハンタイ", "phon": ["h", "a", "N", "t", "a", "i"]},
                   21: {"orth": "失敗", "ヨミ": "シッパイ", "phon": ["sh", "i", "q", "p", "a", "i"]},
                   22: {"orth": "作品", "ヨミ": "サクヒン", "phon": ["s", "a", "k", "u", "h", "i", "N"]},
                   23: {"orth": "指定", "ヨミ": "シテイ", "phon": ["sh", "i", "t", "e", "i"]},
                   24: {"orth": "実験", "ヨミ": "ジッケン", "phon": ["j", "i", "q", "k", "e", "N"]},
                   25: {"orth": "決定", "ヨミ": "ケッテイ", "phon": ["k", "e", "q", "t", "e", "i"]},
                   26: {"orth": "独占", "ヨミ": "ドクセン", "phon": ['d', 'o', 'k', 'u', 's', 'e', 'N']},
                   27: {"orth": "独身", "ヨミ": "ドクシン", "phon": ["d", "o", "k", "u", "sh", "i", "N"]},
                   28: {"orth": "固定", "ヨミ": "コテイ", "phon": ["k", "o", "t", "e", "i"]},
                   29: {"orth": "食品", "ヨミ": "ショクヒン", "phon": ["sh", "o", "k", "u", "h", "i", "N"]},
                   30: {"orth": "表明", "ヨミ": "ヒョウメイ", "phon": ["hy", "o:", "m", "e", "i"]},
                   31: {"orth": "安定", "ヨミ": "アンテイ", "phon": ["a", "N", "t", "e", "i"]},
                   32: {"orth": "各種", "ヨミ": "カクシュ", "phon": ["k", "a", "k", "u", "sh", "u"]},
                   33: {"orth": "役所", "ヨミ": "ヤクショ", "phon": ['y', 'a', 'k', 'u', 'sh', 'o']},
                   34: {"orth": "海岸", "ヨミ": "カイガン", "phon": ["k", "a", "i", "g", "a", "N"]},
                   35: {"orth": "決算", "ヨミ": "ケッサン", "phon": ["k", "e", "q", "s", "a", "N"]},
                   36: {"orth": "地帯", "ヨミ": "チタイ", "phon": ["ch", "i", "t", "a", "i"]},
                   37: {"orth": "道路", "ヨミ": "ドウロ", "phon": ["d", "o:", "r", "o"]},
                   38: {"orth": "安打", "ヨミ": "アンダ", "phon": ["a", "N", "d", "a"]},
                   39: {"orth": "楽団", "ヨミ": "ガクダン", "phon": ["g", "a", "k", "u", "d", "a", "N"]},
                   40: {"orth": "仲間", "ヨミ": "ナカマ", "phon": ["n", "a", "k", "a", "m", "a"]},
                   41: {"orth": "夫婦", "ヨミ": "フウフ", "phon": ["f", "u:", "f", "u"]},
                   42: {"orth": "人間", "ヨミ": "ニンゲン", "phon": ["n", "i", "N", "g", "e", "N"]},
                   43: {"orth": "神経", "ヨミ": "シンケイ", "phon": ["sh", "i", "N", "k", "e", "i"]},
                   44: {"orth": "相手", "ヨミ": "アイテ", "phon": ["a", "i", "t", "e"]},
                   45: {"orth": "反発", "ヨミ": "ハンパツ", "phon": ["h", "a", "N", "p", "a", "ts", "u"]},
                   46: {"orth": "化粧", "ヨミ": "ケショウ", "phon": ["k", "e", "sh", "o:"]},
                   47: {"orth": "建物", "ヨミ": "タテモノ", "phon": ["t", "a", "t", "e", "m", "o", "n", "o"]},
                   48: {"orth": "彼女", "ヨミ": "カノジョ", "phon": ["k", "a", "n", "o", "j", "o"]},
                   49: {"orth": "毛糸", "ヨミ": "ケイト", "phon": ["k", "e", "i", "t", "o"]},
                   50: {"orth": "場合", "ヨミ": "バアイ", "phon": ["b", "a:", "i"]},
                   51: {"orth": "台風", "ヨミ": "タイフウ", "phon": ["t", "a", "i", "f", "u:"]},
                   52: {"orth": "夜間", "ヨミ": "ヤカン", "phon": ["y", "a", "k", "a", "N"]},
                   53: {"orth": "人形", "ヨミ": "ニンギョウ", "phon": ["n", "i", "N", "gy", "o:"]},
                   54: {"orth": "東西", "ヨミ": "トウザイ", "phon": ["t", "o:", "z", "a", "i"]},
                   55: {"orth": "地元", "ヨミ": "ジモト", "phon": ["j", "i", "m", "o", "t", "o"]},
                   56: {"orth": "松原", "ヨミ": "マツバラ", "phon": ["m", "a", "ts", "u", "b", "a", "r", "a"]},
                   57: {"orth": "競馬", "ヨミ": "ケイバ", "phon": ["k", "e", "i", "b", "a"]},
                   58: {"orth": "大幅", "ヨミ": "オオハバ", "phon": ["o:", "h", "a", "b", "a"]},
                   59: {"orth": "貸家", "ヨミ": "カシヤ", "phon": ["k", "a", "sh", "i", "y", "a"]},
                   60: {"orth": "集計", "ヨミ": "シュウケイ", "phon": ["sh", "u:", "k", "e", "i"]},
                   61: {"orth": "観察", "ヨミ": "カンサツ", "phon": ["k", "a", "N", "s", "a", "ts", "u"]},
                   62: {"orth": "予告", "ヨミ": "ヨコク", "phon": ["y", "o", "k", "o", "k", "u"]},
                   63: {"orth": "動脈", "ヨミ": "ドウミャク", "phon": ["d", "o:", "my", "a", "k", "u"]},
                   64: {"orth": "理学", "ヨミ": "リガク", "phon": ["r", "i", "g", "a", "k", "u"]},
                   65: {"orth": "信任", "ヨミ": "シンニン", "phon": ["sh", "i", "N", "n", "i", "N"]},
                   66: {"orth": "任務", "ヨミ": "ニンム", "phon": ["n", "i", "N", "m", "u"]},
                   67: {"orth": "返信", "ヨミ": "ヘンシン", "phon": ["h", "e", "N", "sh", "i", "N"]},
                   68: {"orth": "医局", "ヨミ": "イキョク", "phon": ["i", "ky", "o", "k", "u"]},
                   69: {"orth": "低温", "ヨミ": "テイオン", "phon": ["t", "e", "i", "o", "N"]},
                   70: {"orth": "区別", "ヨミ": "クベツ", "phon": ["k", "u", "b", "e", "ts", "u"]},
                   71: {"orth": "永続", "ヨミ": "エイゾク", "phon": ["e", "i", "z", "o", "k", "u"]},
                   72: {"orth": "持続", "ヨミ": "ジゾク", "phon": ["j", "i", "z", "o", "k", "u"]},
                   73: {"orth": "試練", "ヨミ": "シレン", "phon": ["sh", "i", "r", "e", "N"]},
                   74: {"orth": "満開", "ヨミ": "マンカイ", "phon": ["m", "a", "N", "k", "a", "i"]},
                   75: {"orth": "軍備", "ヨミ": "グンビ", "phon": ["g", "u", "N", "b", "i"]},
                   76: {"orth": "製材", "ヨミ": "セイザイ", "phon": ["s", "e", "i", "z", "a", "i"]},
                   77: {"orth": "銀貨", "ヨミ": "ギンカ", "phon": ["g", "i", "N", "k", "a"]},
                   78: {"orth": "急送", "ヨミ": "キュウソウ", "phon": ["ky", "u:", "s", "o:"]},
                   79: {"orth": "改選", "ヨミ": "カイセン", "phon": ["k", "a", "i", "s", "e", "N"]},
                   80: {"orth": "表紙", "ヨミ": "ヒョウシ", "phon": ["hy", "o:", "sh", "i"]},
                   81: {"orth": "指針", "ヨミ": "シシン", "phon": ["sh", "i", "sh", "i", "N"]},
                   82: {"orth": "熱帯", "ヨミ": "ネッタイ", "phon": ["n", "e", "q", "t", "a", "i"]},
                   83: {"orth": "作詞", "ヨミ": "サクシ", "phon": ["s", "a", "k", "u", "sh", "i"]},
                   84: {"orth": "決着", "ヨミ": "ケッチャク", "phon": ["k", "e", "q", "ch", "a", "k", "u"]},
                   85: {"orth": "食費", "ヨミ": "ショクヒ", "phon": ["sh", "o", "k", "u", "h", "i"]},
                   86: {"orth": "古代", "ヨミ": "コダイ", "phon": ["k", "o", "d", "a", "i"]},
                   87: {"orth": "地形", "ヨミ": "チケイ", "phon": ["ch", "i", "k", "e", "i"]},
                   88: {"orth": "役場", "ヨミ": "ヤクバ", "phon": ["y", "a", "k", "u", "b", "a"]},
                   89: {"orth": "品種", "ヨミ": "ヒンシュ", "phon": ["h", "i", "N", "sh", "u"]},
                   90: {"orth": "祝福", "ヨミ": "シュクフク", "phon": ["sh", "u", "k", "u", "f", "u", "k", "u"]},
                   91: {"orth": "金銭", "ヨミ": "キンセン", "phon": ["k", "i", "N", "s", "e", "N"]},
                   92: {"orth": "根底", "ヨミ": "コンテイ", "phon": ["k", "o", "N", "t", "e", "i"]},
                   93: {"orth": "接種", "ヨミ": "セッシュ", "phon": ["s", "e", "q", "sh", "u"]},
                   94: {"orth": "経由", "ヨミ": "ケイユ", "phon": ["k", "e", "i", "y", "u"]},
                   95: {"orth": "郷土", "ヨミ": "キョウド", "phon": ["ky", "o:", "d", "o"]},
                   96: {"orth": "街路", "ヨミ": "ガイロ", "phon": ["g", "a", "i", "r", "o"]},
                   97: {"orth": "宿直", "ヨミ": "シュクチョク", "phon": ["sh", "u", "k", "u", "ch", "o", "k", "u"]},
                   98: {"orth": "曲折", "ヨミ": "キョクセツ", "phon": ["ky", "o", "k", "u", "s", "e", "ts", "u"]},
                   99: {"orth": "越境", "ヨミ": "エッキョウ", "phon": ["e", "q", "ky", "o:"]},
                   100: {"orth": "強引", "ヨミ": "ゴウイン", "phon": ["g", "o:", "i", "N"]},
                   101: {"orth": "寿命", "ヨミ": "ジュミョウ", "phon": ["j", "u", "my", "o:"]},
                   102: {"orth": "豆腐", "ヨミ": "トウフ", "phon": ["t", "o:", "f", "u"]},
                   103: {"orth": "出前", "ヨミ": "デマエ", "phon": ["d", "e", "m", "a", "e"]},
                   104: {"orth": "歌声", "ヨミ": "ウタゴエ", "phon": ["u", "t", "a", "g", "o", "e"]},
                   105: {"orth": "近道", "ヨミ": "チカミチ", "phon": ["ch", "i", "k", "a", "m", "i", "ch", "i"]},
                   106: {"orth": "間口", "ヨミ": "マグチ", "phon": ["m", "a", "g", "u", "ch", "i"]},
                   107: {"orth": "風物", "ヨミ": "フウブツ", "phon": ["f", "u:", "b", "u", "ts", "u"]},
                   108: {"orth": "面影", "ヨミ": "オモカゲ", "phon": ["o", "m", "o", "k", "a", "g", "e"]},
                   109: {"orth": "眼鏡", "ヨミ": "メガネ", "phon": ["m", "e", "g", "a", "n", "e"]},
                   110: {"orth": "居所", "ヨミ": "イドコロ", "phon": ["i", "d", "o", "k", "o", "r", "o"]},
                   111: {"orth": "献立", "ヨミ": "コンダテ", "phon": ["k", "o", "N", "d", "a", "t", "e"]},
                   112: {"orth": "小雨", "ヨミ": "コサメ", "phon":['k', 'o', 's', 'a', 'm', 'e']},
                   113: {"orth": "毛皮", "ヨミ": "ケガワ", "phon": ["k", "e", "g", "a", "w", "a"]},
                   114: {"orth": "鳥居", "ヨミ": "トリイ", "phon": ["t", "o", "r", "i:"]},
                   115: {"orth": "仲買", "ヨミ": "ナカガイ", "phon": ["n", "a", "k", "a", "g", "a", "i"]},
                   116: {"orth": "頭取", "ヨミ": "トウドリ", "phon": ["t", "o:", "d", "o", "r", "i"]},
                   117: {"orth": "極上", "ヨミ": "ゴクジョウ", "phon": ["g", "o", "k", "u", "j", "o:"]},
                   118: {"orth": "奉行", "ヨミ": "ブギョウ", "phon": ["b", "u", "gy", "o:"]},
                   119: {"orth": "夢路", "ヨミ": "ユメジ", "phon": ["y", "u", "m", "e", "j", "i"]}}

        orth_maxlen, phon_maxlen = 0, 0
        for k, v in data_dict.items():
            wrd = v['orth']
            idx = len(data_dict)
            orth_len = len(v['orth'])
            if orth_len > orth_maxlen:
                orth_maxlen = orth_len
            phon_len = len(v['phon'])
            if phon_len > phon_maxlen:
                phon_maxlen = phon_len

        self.orth_maxlen = orth_maxlen + 1
        self.phon_maxlen = phon_maxlen + 1
        if self.orth_maxlen > self.phon_maxlen:
            self.maxlen = orth_maxlen
        else:
            self.maxlen = self.phon_maxlen

        self.data_dict = data_dict
        orth2info_dict = {}
        for k, v in data_dict.items():
            orth = v['orth']
            orth2info_dict[orth] = v
        self.orth2info_dict = orth2info_dict
        super().set_source_and_target_from_params(source=source, target=target)


    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, x:int, **kwargs):
        _inp = self.data_dict[x][self.source]
        _tch = self.data_dict[x][self.target]
        _inp_ids = self.source_tkn2ids(_inp)
        _tch_ids = self.target_tkn2ids(_tch)
        return _inp_ids + [self.source_list.index('<EOW>')], _tch_ids + [self.target_list.index('<EOW>')]

