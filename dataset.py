import torch

import jaconv
import gzip
import json
import sys
import re

import os
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
        from triangle2023 import char_ja
        return char_ja.kuten().chars

    def get_joyo_chars(self):
        from triangle2023 import chars_joyo as chars_joyo
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

        psylex71_data_fname = 'reading_ja_model/psylex71_data.gz'
        vdrj_data_fname = 'reading_ja_model/vdrj_data.gz'
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

        self.orth_maxlen = orth_maxlen
        self.phon_maxlen = phon_maxlen

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
                 stop_list:list=fushimi1999_list[:120]):

        vdrj_data_fname = 'reading_ja_model/vdrj_data.gz'
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

        self.orth_maxlen = orth_maxlen
        self.phon_maxlen = phon_maxlen

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

        # data_dict の key はインデックス番号とする
        data_dict, orth_maxlen, phon_maxlen = {}, 0, 0
        for wrd in jchar_list[4:]:
            _kata = mecab_yomi(wrd).strip()
            _hira = jaconv.kata2hira(_kata)
            _phon = jaconv.hiragana2julius(_hira).split(' ')
            _idx = len(data_dict)
            #print(f'wrd:{wrd}, _kata:{_kata}, _hira:{_hira}, _phon:{_phon}, _idx:{_idx}')
            phon_len = len(_phon)
            if phon_len > phon_maxlen:
                phon_maxlen

            data_dict[_idx] = {
                'orth': wrd,
                'phon': _phon,
            }

        self.orth_maxlen = 1
        self.phon_maxlen = phon_maxlen
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
