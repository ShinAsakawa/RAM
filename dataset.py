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
                 max_words:int=200000):
        super().__init__()

        self.jchar_list = self.get_jchar_list()
        self.phon_list = self.get_phon_list()
        self.joyo_list = self.get_joyo_chars()
        self.source=source
        self.target=target
        self.yomi=mecab_yomi

        self.orth_list = self.joyo_list
        self.max_words = max_words

        #self.set_source_and_target_from_params(source=source, target=target)

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

    def __getitem__(self, **kwargs):
        """ Must be overrided """
        raise NotImplementedError

    def set_source_and_target_from_params(self, source:str='orth', target:str='phon'):
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

        super().__init__()
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

        super().__init__()
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
                 #mecab_yomi=mecab_yomi,
                 stop_list:list=None):

        super().__init__()
        self.datasetname = 'onechar'
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
        #self.mecab_yomi = mecab_yomi

        # data_dict の key はインデックス番号とする
        # data_dict, orth_maxlen, phon_maxlen = {}, 0, 0
        # for wrd in jchar_list[4:]:
        #     _kata = mecab_yomi(wrd).strip()
        #     _phon = self.kata2phon(_kata)
        #     _idx = len(data_dict)
        #     phon_len = len(_phon)
        #     if phon_len > phon_maxlen:
        #         phon_maxlen = phon_len

        #     data_dict[_idx] = {
        #         'orth': wrd,
        #         'phon': _phon,
        #     }

        data_dict = {
            0: {'orth': '０', 'phon': ['z', 'e', 'r', 'o']},
            1: {'orth': '１', 'phon': ['i', 'ch', 'i']},
            2: {'orth': '２', 'phon': ['n', 'i']},
            3: {'orth': '３', 'phon': ['s', 'a', 'N']},
            4: {'orth': '４', 'phon': ['y', 'o', 'N']},
            5: {'orth': '５', 'phon': ['g', 'o']},
            6: {'orth': '６', 'phon': ['r', 'o', 'k', 'u']},
            7: {'orth': '７', 'phon': ['n', 'a', 'n', 'a']},
            8: {'orth': '８', 'phon': ['h', 'a', 'ch', 'i']},
            9: {'orth': '９', 'phon': ['ky', 'u:']},
            10: {'orth': 'Ａ', 'phon': ['e', 'i']},
            11: {'orth': 'Ｂ', 'phon': ['b', 'i:']},
            12: {'orth': 'Ｃ', 'phon': ['sh', 'i:']},
            13: {'orth': 'Ｄ', 'phon': ['d', 'i:']},
            14: {'orth': 'Ｅ', 'phon': ['i:']},
            15: {'orth': 'Ｆ', 'phon': ['e', 'f', 'u']},
            16: {'orth': 'Ｇ', 'phon': ['j', 'i:']},
            17: {'orth': 'Ｈ', 'phon': ['e', 'i', 'ch', 'i']},
            18: {'orth': 'Ｉ', 'phon': ['a', 'i']},
            19: {'orth': 'Ｊ', 'phon': ['j', 'e', 'i']},
            20: {'orth': 'Ｋ', 'phon': ['k', 'e', 'i']},
            21: {'orth': 'Ｌ', 'phon': ['e', 'r', 'u']},
            22: {'orth': 'Ｍ', 'phon': ['e', 'm', 'u']},
            23: {'orth': 'Ｎ', 'phon': ['e', 'n', 'u']},
            24: {'orth': 'Ｏ', 'phon': ['o:']},
            25: {'orth': 'Ｐ', 'phon': ['p', 'i:']},
            26: {'orth': 'Ｑ', 'phon': ['ky', 'u:']},
            27: {'orth': 'Ｒ', 'phon': ['a:', 'r', 'u']},
            28: {'orth': 'Ｓ', 'phon': ['e', 's', 'u']},
            29: {'orth': 'Ｔ', 'phon': ['t', 'i:']},
            30: {'orth': 'Ｕ', 'phon': ['y', 'u:']},
            31: {'orth': 'Ｖ', 'phon': ['b', 'u', 'i']},
            32: {'orth': 'Ｗ', 'phon': ['d', 'a', 'b', 'u', 'ry', 'u:']},
            33: {'orth': 'Ｘ', 'phon': ['e', 'q', 'k', 'u', 's', 'u']},
            34: {'orth': 'Ｙ', 'phon': ['w', 'a', 'i']},
            35: {'orth': 'Ｚ', 'phon': ['z', 'i:']},
            36: {'orth': 'ぁ', 'phon': ['a']},
            37: {'orth': 'あ', 'phon': ['a']},
            38: {'orth': 'ぃ', 'phon': ['i']},
            39: {'orth': 'い', 'phon': ['i']},
            40: {'orth': 'ぅ', 'phon': ['u']},
            41: {'orth': 'う', 'phon': ['u']},
            42: {'orth': 'ぇ', 'phon': ['e']},
            43: {'orth': 'え', 'phon': ['e']},
            44: {'orth': 'ぉ', 'phon': ['o']},
            45: {'orth': 'お', 'phon': ['o']},
            46: {'orth': 'か', 'phon': ['k', 'a']},
            47: {'orth': 'が', 'phon': ['g', 'a']},
            48: {'orth': 'き', 'phon': ['k', 'i']},
            49: {'orth': 'ぎ', 'phon': ['g', 'i']},
            50: {'orth': 'く', 'phon': ['k', 'u']},
            51: {'orth': 'ぐ', 'phon': ['g', 'u']},
            52: {'orth': 'け', 'phon': ['k', 'e']},
            53: {'orth': 'げ', 'phon': ['g', 'e']},
            54: {'orth': 'こ', 'phon': ['k', 'o']},
            55: {'orth': 'ご', 'phon': ['g', 'o']},
            56: {'orth': 'さ', 'phon': ['s', 'a']},
            57: {'orth': 'ざ', 'phon': ['z', 'a']},
            58: {'orth': 'し', 'phon': ['sh', 'i']},
            59: {'orth': 'じ', 'phon': ['j', 'i']},
            60: {'orth': 'す', 'phon': ['s', 'u']},
            61: {'orth': 'ず', 'phon': ['z', 'u']},
            62: {'orth': 'せ', 'phon': ['s', 'e']},
            63: {'orth': 'ぜ', 'phon': ['z', 'e']},
            64: {'orth': 'そ', 'phon': ['s', 'o']},
            65: {'orth': 'ぞ', 'phon': ['z', 'o']},
            66: {'orth': 'た', 'phon': ['t', 'a']},
            67: {'orth': 'だ', 'phon': ['d', 'a']},
            68: {'orth': 'ち', 'phon': ['ch', 'i']},
            69: {'orth': 'ぢ', 'phon': ['j', 'i']},
            70: {'orth': 'っ', 'phon': ['q']},
            71: {'orth': 'つ', 'phon': ['ts', 'u']},
            72: {'orth': 'づ', 'phon': ['z', 'u']},
            73: {'orth': 'て', 'phon': ['t', 'e']},
            74: {'orth': 'で', 'phon': ['d', 'e']},
            75: {'orth': 'と', 'phon': ['t', 'o']},
            76: {'orth': 'ど', 'phon': ['d', 'o']},
            77: {'orth': 'な', 'phon': ['n', 'a']},
            78: {'orth': 'に', 'phon': ['n', 'i']},
            79: {'orth': 'ぬ', 'phon': ['n', 'u']},
            80: {'orth': 'ね', 'phon': ['n', 'e']},
            81: {'orth': 'の', 'phon': ['n', 'o']},
            82: {'orth': 'は', 'phon': ['h', 'a']},
            83: {'orth': 'ば', 'phon': ['b', 'a']},
            84: {'orth': 'ぱ', 'phon': ['p', 'a']},
            85: {'orth': 'ひ', 'phon': ['h', 'i']},
            86: {'orth': 'び', 'phon': ['b', 'i']},
            87: {'orth': 'ぴ', 'phon': ['p', 'i']},
            88: {'orth': 'ふ', 'phon': ['f', 'u']},
            89: {'orth': 'ぶ', 'phon': ['b', 'u']},
            90: {'orth': 'ぷ', 'phon': ['p', 'u']},
            91: {'orth': 'へ', 'phon': ['h', 'e']},
            92: {'orth': 'べ', 'phon': ['b', 'e']},
            93: {'orth': 'ぺ', 'phon': ['p', 'e']},
            94: {'orth': 'ほ', 'phon': ['h', 'o']},
            95: {'orth': 'ぼ', 'phon': ['b', 'o']},
            96: {'orth': 'ぽ', 'phon': ['p', 'o']},
            97: {'orth': 'ま', 'phon': ['m', 'a']},
            98: {'orth': 'み', 'phon': ['m', 'i']},
            99: {'orth': 'む', 'phon': ['m', 'u']},
            100: {'orth': 'め', 'phon': ['m', 'e']},
            101: {'orth': 'も', 'phon': ['m', 'o']},
            102: {'orth': 'ゃ', 'phon': ['y', 'a']},
            103: {'orth': 'や', 'phon': ['y', 'a']},
            104: {'orth': 'ゅ', 'phon': ['y', 'y']},
            105: {'orth': 'ゆ', 'phon': ['y', 'u']},
            106: {'orth': 'ょ', 'phon': ['y', 'o']},
            107: {'orth': 'よ', 'phon': ['y', 'o']},
            108: {'orth': 'ら', 'phon': ['r', 'a']},
            109: {'orth': 'り', 'phon': ['r', 'i']},
            110: {'orth': 'る', 'phon': ['r', 'u']},
            111: {'orth': 'れ', 'phon': ['r', 'e']},
            112: {'orth': 'ろ', 'phon': ['r', 'o']},
            113: {'orth': 'ゎ', 'phon': ['w', 'a']},
            114: {'orth': 'わ', 'phon': ['w', 'a']},
            115: {'orth': 'ゐ', 'phon': ['i']},
            116: {'orth': 'ゑ', 'phon': ['e']},
            117: {'orth': 'を', 'phon': ['o']},
            118: {'orth': 'ん', 'phon': ['N']}}

        orth_maxlen, phon_maxlen = 1, 6
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

        super().__init__()
        self.datasetname = 'fushimi1999'

        self.max_words = max_words
        self.phon_list = super().get_phon_list()
        self.jchar_list = super().get_jchar_list()
        self.joyo_chars = super().get_joyo_chars()
        self.source = source
        self.target = target
        self.orth_list = self.jchar_list

        self.conds = {
            # consistent, high-frequency words
            'HF___consist__': ['戦争', '倉庫', '医学', '注意', '記念', '番号', '料理', '完全', '開始', '印刷', '連続', '予約', '多少', '教員', '当局', '材料', '夕刊', '労働', '運送', '電池'],
            # inconsistent, high-frequency words
            'HF___inconsist': ['反対', '失敗', '作品', '指定', '実験', '決定', '独占', '独身', '固定', '食品', '表明', '安定', '各種', '役所', '海岸', '決算', '地帯', '道路', '安打', '楽団'],
            # inconsistent atypical, 'high-frequency words
            'HF___atypical_': ['仲間', '夫婦', '人間', '神経', '相手', '反発', '化粧', '建物', '彼女', '毛糸', '場合', '台風', '夜間', '人形', '東西', '地元', '松原', '競馬', '大幅', '貸家'],
            # consistent, low-frequency words
            'LF___consist__': ['集計', '観察', '予告', '動脈', '理学', '信任', '任務', '返信', '医局', '低温', '区別', '永続', '持続', '試練', '満開', '軍備', '製材', '銀貨', '急送', '改選'],
            # inconsistent, low-frequency words
            'LF___inconsist': ['表紙', '指針', '熱帯', '作詞', '決着', '食費', '古代', '地形', '役場', '品種', '祝福', '金銭', '根底', '接種', '経由', '郷土', '街路', '宿直', '曲折', '越境'],
            # inconsistent atypical, low-frequency words
            'LF___atypical_': ['強引', '寿命', '豆腐', '出前', '歌声', '近道', '間口', '風物', '面影', '眼鏡', '居所', '献立', '小雨', '毛皮', '鳥居', '仲買', '頭取', '極上', '奉行', '夢路'],
            # consistent, 'high-character-frequency non-words
            'HFNW_consist__': ['集学', '信別', '製信', '運学', '番送', '電続', '完意', '軍開', '動選', '当働', '予続', '倉理', '予少', '教池', '理任', '銀務', '連料', '開員', '注全', '記争'],
            # inconsistent biased, 'high-character-frequency non-words
            'HFNW_inconsist': ['作明', '風行', '失定', '指団', '決所', '各算', '海身', '東発', '楽験', '作代', '反原', '独対', '歌上', '反定', '独定', '場家', '安種', '経着', '決土', '松合'],
            # inconsistent ambiguous, 'high-character-frequency non-words
            'HFNW_ambiguous': ['表品', '実定', '人風', '神間', '相経', '人元', '小引', '指場', '毛所', '台手', '間物', '道品', '出取', '建馬', '大婦', '地打', '化間', '面口', '金由', '彼間'],
            # consistent, 'low-character-frequency non-words
            'LFNW_consist__': ['急材', '戦刊', '返計', '印念', '低局', '労号', '満送', '永告', '試脈', '観備', '材約', '夕局', '医庫', '任続', '医貨', '改練', '区温', '多始', '材刷', '持察'],
            # inconsistent biased, 'low-character-frequency non-words
            'LFNW_inconsist': ['食占', '表底', '宿帯', '決帯', '古費', '安敗', '役針', '近命', '眼道', '豆立', '街直', '固路', '郷種', '品路', '曲銭', '献居', '奉買', '根境', '役岸', '祝折'],
            # inconsistent ambiguous, 'low-character-frequency non-words
            'LFNW_ambiguous': ['食形', '接紙', '競物', '地詞', '強腐', '頭路', '毛西', '夜糸', '仲影', '熱福', '寿前', '鳥雨', '地粧', '越種', '仲女', '極鏡', '夢皮', '居声', '貸形', '夫幅']}

        _wrds = []
        for k in self.conds.keys():
            for ch in self.conds[k]:
                _wrds.append(ch)
        self.all_words = _wrds
        self.real_words = _wrds[:120]

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


class TLPA_Dataset(RAM_Dataset):

    def __init__(self,
                 source:str="orth",
                 target:str="phon",
                 max_words:int=None,
                 task:str="tlpa1",   # ['tlpa1', 'tlpa2', 'tlpa3', 'tlpa4']
                 ):

        super().__init__()

        self.task = task
        self.phon_list = super().get_phon_list()
        self.jchar_list = super().get_jchar_list()
        self.joyo_chars = super().get_joyo_chars()
        self.source = source
        self.target = target
        self.orth_list = self.jchar_list

        self.tlpa1 = {
            0: {'orth': '学生', 'phon': ['g', 'a', 'k', 'u', 's', 'e', 'i'], 'cond': 'HH', 'ヨミ': 'ガクセイ'},
            1: {'orth': '基調', 'phon': ['k', 'i', 'ch', 'o:'], 'cond': 'LL', 'ヨミ': 'キチョウ'},
            2: {'orth': '冷屋', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            3: {'orth': '制限', 'phon': ['s', 'e', 'i', 'g', 'e', 'N'],  'cond': 'HL',  'ヨミ': 'セイゲン'},
            4: {'orth': '外士', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            5: {'orth': '主草', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            6: {'orth': '自重', 'phon': ['j', 'i', 'ch', 'o:'], 'cond': 'LL', 'ヨミ': 'ジチョウ'},
            7: {'orth': '生海', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            8: {'orth': '音博', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            9: {'orth': '当時', 'phon': ['t', 'o:', 'j', 'i'], 'cond': 'HL', 'ヨミ': 'トウジ'},
            10: {'orth': '想録', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            11: {'orth': '常益', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            12: {'orth': '手術', 'phon': ['sh', 'u', 'j', 'u', 'ts', 'u'], 'cond': 'HH', 'ヨミ': 'シュジュツ'},
            13: {'orth': '逆転', 'phon': ['gy', 'a', 'k', 'u', 't', 'e', 'N'], 'cond': 'LL', 'ヨミ': 'ギャクテン'},
            14: {'orth': '金属', 'phon': ['k', 'i', 'N', 'z', 'o', 'k', 'u'],  'cond': 'HH',  'ヨミ': 'キンゾク'},
            15: {'orth': '巻器', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            16: {'orth': '戦争', 'phon': ['s', 'e', 'N', 's', 'o:'], 'cond': 'HH', 'ヨミ': 'センソウ'},
            17: {'orth': '識文', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            18: {'orth': '車園', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            19: {'orth': '田園', 'phon': ['d', 'e', 'N', 'e', 'N'], 'cond': 'LH', 'ヨミ': 'デンエン'},
            20: {'orth': '状態', 'phon': ['j', 'o:', 't', 'a', 'i'], 'cond': 'HL', 'ヨミ': 'ジョウタイ'},
            21: {'orth': '水泳', 'phon': ['s', 'u', 'i', 'e', 'i'], 'cond': 'LH', 'ヨミ': 'スイエイ'},
            22: {'orth': '続基', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            23: {'orth': '通学', 'phon': ['ts', 'u:', 'g', 'a', 'k', 'u'], 'cond': 'LH', 'ヨミ': 'ツウガク'},
            24: {'orth': '福供', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            25: {'orth': '発想', 'phon': ['h', 'a', 'q', 's', 'o:'],  'cond': 'LL',  'ヨミ': 'ハッソウ'},
            26: {'orth': '永調', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            27: {'orth': '夫婦', 'phon': ['f', 'u:', 'f', 'u'], 'cond': 'HH', 'ヨミ': 'フウフ'},
            28: {'orth': '結果', 'phon': ['k', 'e', 'q', 'k', 'a'],  'cond': 'HL', 'ヨミ': 'ケッカ'},
            29: {'orth': '具血', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            30: {'orth': '自安', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            31: {'orth': '思想', 'phon': ['sh', 'i', 's', 'o:'], 'cond': 'HL', 'ヨミ': 'シソウ'},
            32: {'orth': '選定', 'phon': ['s', 'e', 'N', 't', 'e', 'i'], 'cond': 'LL',  'ヨミ': 'センテイ'},
            33: {'orth': '果樹', 'phon': ['k', 'a', 'j', 'u'], 'cond': 'LH', 'ヨミ': 'カジュ'},
            34: {'orth': '職価', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            35: {'orth': '関係', 'phon': ['k', 'a', 'N', 'k', 'e', 'i'], 'cond': 'HL', 'ヨミ': 'カンケイ'},
            36: {'orth': '信用', 'phon': ['sh', 'i', 'N', 'y', 'o:'],  'cond': 'HL',  'ヨミ': 'シンヨウ'},
            37: {'orth': '済囲', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            38: {'orth': '講製', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            39: {'orth': '当態', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            40: {'orth': '放送', 'phon': ['h', 'o:', 's', 'o:'], 'cond': 'HH', 'ヨミ': 'ホウソウ'},
            41: {'orth': '海外', 'phon': ['k', 'a', 'i', 'g', 'a', 'i'],  'cond': 'HH', 'ヨミ': 'カイガイ'},
            42: {'orth': '発収', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            43: {'orth': '種類', 'phon': ['sh', 'u', 'r', 'u', 'i'], 'cond': 'HL', 'ヨミ': 'シュルイ'},
            44: {'orth': '温泉', 'phon': ['o', 'N', 's', 'e', 'N'], 'cond': 'HH', 'ヨミ': 'オンセン'},
            45: {'orth': '所学', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            46: {'orth': '車楽', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            47: {'orth': '制化', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            48: {'orth': '控室', 'phon': ['h', 'i', 'k', 'a', 'e', 'sh', 'i', 'ts', 'u'],  'cond': 'LH',  'ヨミ': 'ヒカエシツ'},
            49: {'orth': '場花', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            50: {'orth': '役火', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            51: {'orth': '収益', 'phon': ['sh', 'u:', 'e', 'k', 'i'],  'cond': 'HL',  'ヨミ': 'シュウエキ'},
            52: {'orth': '泳田', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            53: {'orth': '定外', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            54: {'orth': '船長', 'phon': ['s', 'e', 'N', 'ch', 'o:'],  'cond': 'LH',  'ヨミ': 'センチョウ'},
            55: {'orth': '演幸', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            56: {'orth': '親戚', 'phon': ['sh', 'i', 'N', 's', 'e', 'k', 'i'],  'cond': 'LH',  'ヨミ': 'シンセキ'},
            57: {'orth': '精製', 'phon': ['s', 'e', 'i', 's', 'e', 'i'],  'cond': 'LL', 'ヨミ': 'セイセイ'},
            58: {'orth': '察手', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            59: {'orth': '冷凍', 'phon': ['r', 'e', 'i', 't', 'o:'],  'cond': 'LH', 'ヨミ': 'レイトウ'},
            60: {'orth': '経済', 'phon': ['k', 'e', 'i', 'z', 'a', 'i'], 'cond': 'HL',  'ヨミ': 'ケイザイ'},
            61: {'orth': '水内', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            62: {'orth': '気思', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            63: {'orth': '夫送', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            64: {'orth': '講和', 'phon': ['k', 'o:', 'w', 'a'], 'cond': 'LL', 'ヨミ': 'コウワ'},
            65: {'orth': '病温', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            66: {'orth': '幸福', 'phon': ['k', 'o:', 'f', 'u', 'k', 'u'], 'cond': 'HH', 'ヨミ': 'コウフク'},
            67: {'orth': '笑長', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            68: {'orth': '種収', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            69: {'orth': '味想', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            70: {'orth': '収録', 'phon': ['sh', 'u:', 'r', 'o', 'k', 'u'],  'cond': 'LL',  'ヨミ': 'シュウロク'},
            71: {'orth': '役所', 'phon': ['y', 'a', 'k', 'u', 'sh', 'o'],  'cond': 'LH', 'ヨミ': 'ヤクショ'},
            72: {'orth': '主元', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            73: {'orth': '保安', 'phon': ['h', 'o', 'a', 'N'], 'cond': 'LL', 'ヨミ': 'ホアン'},
            74: {'orth': '職権', 'phon': ['sh', 'o', 'q', 'k', 'e', 'N'], 'cond': 'LL', 'ヨミ': 'ショッケン'},
            75: {'orth': '写真', 'phon': ['sh', 'a', 'sh', 'i', 'N'], 'cond': 'HH', 'ヨミ': 'シャシン'},
            76: {'orth': '前類', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            77: {'orth': '望閉', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            78: {'orth': '場列', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            79: {'orth': '屋上', 'phon': ['o', 'k', 'u', 'j', 'o:'], 'cond': 'LH', 'ヨミ': 'オクジョウ'},
            80: {'orth': '必信', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            81: {'orth': '条項', 'phon': ['j', 'o:', 'k', 'o:'], 'cond': 'LL', 'ヨミ': 'ジョウコウ'},
            82: {'orth': '周囲', 'phon': ['sh', 'u:', 'i'], 'cond': 'HL', 'ヨミ': 'シュウイ'},
            83: {'orth': '説婦', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            84: {'orth': '意結', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            85: {'orth': '選案', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            86: {'orth': '葉巻', 'phon': ['h', 'a', 'm', 'a', 'k', 'i'], 'cond': 'LH', 'ヨミ': 'ハマキ'},
            87: {'orth': '状個', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            88: {'orth': '協経', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            89: {'orth': '容器', 'phon': ['y', 'o:', 'k', 'i'], 'cond': 'LH', 'ヨミ': 'ヨウキ'},
            90: {'orth': '子供', 'phon': ['k', 'o', 'd', 'o', 'm', 'o'], 'cond': 'HH', 'ヨミ': 'コドモ'},
            91: {'orth': '学真', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            92: {'orth': '草案', 'phon': ['s', 'o:', 'a', 'N'], 'cond': 'LL', 'ヨミ': 'ソウアン'},
            93: {'orth': '外観', 'phon': ['g', 'a', 'i', 'k', 'a', 'N'], 'cond': 'LLL', 'ヨミ': 'ガイカン'},
            94: {'orth': '主眼', 'phon': ['sh', 'u', 'g', 'a', 'N'], 'cond': 'LL', 'ヨミ': 'シュガン'},
            95: {'orth': '属写', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            96: {'orth': '保判', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            97: {'orth': '必要', 'phon': ['h', 'i', 'ts', 'u', 'y', 'o:'], 'cond': 'HL',  'ヨミ': 'ヒツヨウ'},
            98: {'orth': '音楽', 'phon': ['o', 'N', 'g', 'a', 'k', 'u'], 'cond': 'HH', 'ヨミ': 'オンガク'},
            99: {'orth': '火花', 'phon': ['h', 'i', 'b', 'a', 'n', 'a'], 'cond': 'LH', 'ヨミ': 'ヒバナ'},
            100: {'orth': '限要', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            101: {'orth': '項持', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            102: {'orth': '文化', 'phon': ['b', 'u', 'N', 'k', 'a'], 'cond': 'HL', 'ヨミ': 'ブンカ'},
            103: {'orth': '演奏', 'phon': ['e', 'N', 's', 'o:'], 'cond': 'HH', 'ヨミ': 'エンソウ'},
            104: {'orth': '口逆', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            105: {'orth': '協定', 'phon': ['ky', 'o:', 't', 'e', 'i'], 'cond': 'HL', 'ヨミ': 'キョウテイ'},
            106: {'orth': '原眼', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            107: {'orth': '小鳥', 'phon': ['k', 'o', 't', 'o', 'r', 'i'], 'cond': 'LH', 'ヨミ': 'コトリ'},
            108: {'orth': '公観', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            109: {'orth': '警察', 'phon': ['k', 'e', 'i', 's', 'a', 'ts', 'u'], 'cond': 'HH', 'ヨミ': 'ケイサツ'},
            110: {'orth': '売場', 'phon': ['u', 'r', 'i', 'b', 'a'], 'cond': 'LH', 'ヨミ': 'ウリバ'},
            111: {'orth': '永続', 'phon': ['e', 'i', 'z', 'o', 'k', 'u'], 'cond': 'LL', 'ヨミ': 'エイゾク'},
            112: {'orth': '重和', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            113: {'orth': '時係', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            114: {'orth': '絵凍', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            115: {'orth': '原価', 'phon': ['g', 'e', 'N', 'k', 'a'], 'cond': 'LL', 'ヨミ': 'ゲンカ'},
            116: {'orth': '公平', 'phon': ['k', 'o:', 'h', 'e', 'i'], 'cond': 'LL', 'ヨミ': 'コウヘイ'},
            117: {'orth': '劇屋', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            118: {'orth': '戚小', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            119: {'orth': '個人', 'phon': ['k', 'o', 'j', 'i', 'N'], 'cond': 'HL', 'ヨミ': 'コジン'},
            120: {'orth': '鳥舟', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            121: {'orth': '常識', 'phon': ['j', 'o:', 'sh', 'i', 'k', 'i'], 'cond': 'HL', 'ヨミ': 'ジョウシキ'},
            122: {'orth': '人関', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            123: {'orth': '判例', 'phon': ['h', 'a', 'N', 'r', 'e', 'i'], 'cond': 'LL', 'ヨミ': 'ハンレイ'},
            124: {'orth': '博士', 'phon': ['h', 'a', 'k', 'a', 's', 'e'], 'cond': 'HH', 'ヨミ': 'ハカセ'},
            125: {'orth': '主義', 'phon': ['sh', 'u', 'g', 'i'], 'cond': 'HL', 'ヨミ': 'シュギ'},
            126: {'orth': '精平', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            127: {'orth': '出血', 'phon': ['sh', 'u', 'q', 'k', 'e', 'ts', 'u'], 'cond': 'LH', 'ヨミ': 'シュッケツ'},
            128: {'orth': '転条', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            129: {'orth': '小説', 'phon': ['sh', 'o:', 's', 'e', 'ts', 'u'], 'cond': 'HH', 'ヨミ': 'ショウセツ'},
            130: {'orth': '爆笑', 'phon': ['b', 'a', 'k', 'u', 'sh', 'o:'], 'cond': 'LH', 'ヨミ': 'バクショウ'},
            131: {'orth': '元気', 'phon': ['g', 'e', 'N', 'k', 'i'], 'cond': 'HL', 'ヨミ': 'ゲンキ'},
            132: {'orth': '戦放', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            133: {'orth': '通室', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            134: {'orth': '気子', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            135: {'orth': '意味', 'phon': ['i', 'm', 'i'], 'cond': 'HL', 'ヨミ': 'イミ'},
            136: {'orth': '爆控', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            137: {'orth': '病気', 'phon': ['by', 'o:', 'k', 'i'], 'cond': 'HH', 'ヨミ': 'ビョウキ'},
            138: {'orth': '車内', 'phon': ['sh', 'a', 'n', 'a', 'i'], 'cond': 'LH', 'ヨミ': 'シャナイ'},
            139: {'orth': '後定', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            140: {'orth': '奏術', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            141: {'orth': '用果', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            142: {'orth': '絵具', 'phon': ['e', 'n', 'o', 'g', 'u'], 'cond': 'LH', 'ヨミ': 'エノグ'},
            143: {'orth': '果出', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            144: {'orth': '閉口', 'phon': ['h', 'e', 'i', 'k', 'o:'], 'cond': 'LL', 'ヨミ': 'ヘイコウ'},
            145: {'orth': '部屋', 'phon': ['h', 'e', 'y', 'a'], 'cond': 'HH', 'ヨミ': 'ヘヤ'},
            146: {'orth': '警小', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            147: {'orth': '劇場', 'phon': ['g', 'e', 'k', 'i', 'j', 'o:'], 'cond': 'HH', 'ヨミ': 'ゲキジョウ'},
            148: {'orth': '前後', 'phon': ['z', 'e', 'N', 'g', 'o'], 'cond': 'HL', 'ヨミ': 'ゼンゴ'},
            149: {'orth': '体葉', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            150: {'orth': '列車', 'phon': ['r', 'e', 'q', 'sh', 'a'], 'cond': 'HH', 'ヨミ': 'レッシャ'},
            151: {'orth': '親売', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            152: {'orth': '周義', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            153: {'orth': '容上', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            154: {'orth': '人体', 'phon': ['j', 'i', 'N', 't', 'a', 'i'], 'cond': 'LH', 'ヨミ': 'ジンタイ'},
            155: {'orth': '金争', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            156: {'orth': '権例', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            157: {'orth': '人樹', 'phon': [], 'cond': 'NW', 'ヨミ': ''},
            158: {'orth': '待望', 'phon': ['t', 'a', 'i', 'b', 'o:'], 'cond': 'LL', 'ヨミ': 'タイボウ'},
            159: {'orth': '部泉', 'phon': [], 'cond': 'NW', 'ヨミ': ''}}

        self.tlpa2 = {
            0:{'orth': 'こより', 'note': '', 'ヨミ': 'コヨリ', 'phon': ['k', 'o', 'y', 'o', 'r', 'i']},
            1:{'orth': 'よもい', 'note': '（よろい）', 'ヨミ': 'ヨモイ', 'phon': ['y', 'o', 'm', 'o', 'i']},
            2:{'orth': 'ゆのみ', 'note': '', 'ヨミ': 'ユノミ', 'phon': ['y', 'u', 'n', 'o', 'm', 'i']},
            3:{'orth': 'たきぎ', 'note': '', 'ヨミ': 'タキギ', 'phon': ['t', 'a', 'k', 'i', 'g', 'i']},
            4:{'orth': 'あつび', 'note': '（あくび）', 'ヨミ': 'アツび', 'phon': ['a', 'ts', 'u', 'b', 'i']},
            5:{'orth': 'けむに', 'note': ' (けむり)', 'ヨミ': 'ケムニ', 'phon': ['k', 'e', 'm', 'u', 'n', 'i']},
            6:{'orth': 'わらじ', 'note': '', 'ヨミ': 'ワラジ', 'phon': ['w', 'a', 'r', 'a', 'j', 'i']},
            7:{'orth': 'いとろ', 'note': '（いとこ）', 'ヨミ': 'イトロ', 'phon': ['i', 't', 'o', 'r', 'o']},
            8:{'orth': 'みきん', 'note': '（みりん）', 'ヨミ': 'ミキン', 'phon': ['m', 'i', 'k', 'i', 'N']},
            9:{'orth': 'なだけ', 'note': '（なだれ）', 'ヨミ': 'ナダケ', 'phon': ['n', 'a', 'd', 'a', 'k', 'e']},
            10:{'orth': 'かがみ', 'note': '', 'ヨミ': 'カガミ', 'phon': ['k', 'a', 'g', 'a', 'm', 'i']},
            11:{'orth': 'もさか', 'note': '（とさか）', 'ヨミ': 'モサカ', 'phon': ['m', 'o', 's', 'a', 'k', 'a']},
            12:{'orth': 'あくび', 'note': '', 'ヨミ': 'アクビ', 'phon': ['a', 'k', 'u', 'b', 'i']},
            13:{'orth': 'よぶし', 'note': '（こぶし）', 'ヨミ': 'ヨブシ', 'phon': ['y', 'o', 'b', 'u', 'sh', 'i']},
            14:{'orth': 'たがみ', 'note': '（かがみ）', 'ヨミ': 'タガミ', 'phon': ['t', 'a', 'g', 'a', 'm', 'i']},
            15:{'orth': 'あらし', 'note': '', 'ヨミ': 'アラシ', 'phon': ['a', 'r', 'a', 'sh', 'i']},
            16:{'orth': 'みりん', 'note': '', 'ヨミ': 'ミリン', 'phon': ['m', 'i', 'r', 'i', 'N']},
            17:{'orth': 'とさか', 'note': '', 'ヨミ': 'トサカ', 'phon': ['t', 'o', 's', 'a', 'k', 'a']},
            18:{'orth': 'わぶと', 'note': '（かぶと）', 'ヨミ': 'ワブト', 'phon': ['w', 'a', 'b', 'u', 't', 'o']},
            19:{'orth': 'けむり', 'note': '', 'ヨミ': 'ケムリ', 'phon': ['k', 'e', 'm', 'u', 'r', 'i']},
            20:{'orth': 'あまし', 'note': '（あらし）', 'ヨミ': 'アマシ', 'phon': ['a', 'm', 'a', 'sh', 'i']},
            21:{'orth': 'うどん', 'note': '', 'ヨミ': 'ウドン', 'phon': ['u', 'd', 'o', 'N']},
            22:{'orth': 'こよち', 'note': '（こより）', 'ヨミ': 'コヨチ', 'phon': ['k', 'o', 'y', 'o', 'ch', 'i']},
            23:{'orth': 'のれん', 'note': '', 'ヨミ': 'ノレン', 'phon': ['n', 'o', 'r', 'e', 'N']},
            24:{'orth': 'むしろ', 'note': '', 'ヨミ': 'ムシロ', 'phon': ['m', 'u', 'sh', 'i', 'r', 'o']},
            25:{'orth': 'やぶら', 'note': '（やぐら）', 'ヨミ': 'ヤブラ', 'phon': ['y', 'a', 'b', 'u', 'r', 'a']},
            26:{'orth': 'なだれ', 'note': '', 'ヨミ': 'ナダレ', 'phon': ['n', 'a', 'd', 'a', 'r', 'e']},
            27:{'orth': 'うごん', 'note': ' (うどん)', 'ヨミ': 'ウゴン', 'phon': ['u', 'g', 'o', 'N']},
            28:{'orth': 'たきり', 'note': ' (たきぎ)', 'ヨミ': 'タキリ', 'phon': ['t', 'a', 'k', 'i', 'r', 'i']},
            29:{'orth': 'いとこ', 'note': '', 'ヨミ': 'イトコ', 'phon': ['i', 't', 'o', 'k', 'o']},
            30:{'orth': 'うちわ', 'note': '', 'ヨミ': 'ウチワ', 'phon': ['u', 'ch', 'i', 'w', 'a']},
            31:{'orth': 'たらじ', 'note': '（わらじ）', 'ヨミ': 'タラジ', 'phon': ['t', 'a', 'r', 'a', 'j', 'i']},
            32:{'orth': 'くのみ', 'note': '（ゆのみ）', 'ヨミ': 'クノミ', 'phon': ['k', 'u', 'n', 'o', 'm', 'i']},
            33:{'orth': 'ゆしろ', 'note': '（むしろ）', 'ヨミ': 'ユシロ', 'phon': ['y', 'u', 'sh', 'i', 'r', 'o']},
            34:{'orth': 'かぶと', 'note': '', 'ヨミ': 'カブト', 'phon': ['k', 'a', 'b', 'u', 't', 'o']},
            35:{'orth': 'こぶし', 'note': '', 'ヨミ': 'コブシ', 'phon': ['k', 'o', 'b', 'u', 'sh', 'i']},
            36:{'orth': 'よろい', 'note': '', 'ヨミ': 'ヨロイ', 'phon': ['y', 'o', 'r', 'o', 'i']},
            37:{'orth': 'うちな', 'note': '（うちわ）', 'ヨミ': 'ウチナ', 'phon': ['u', 'ch', 'i', 'n', 'a']},
            38:{'orth': 'やぐら', 'note': '', 'ヨミ': 'ヤぐら', 'phon': ['y', 'a', 'g', 'u', 'r', 'a']},
            39:{'orth': 'のけん', 'note': '（のれん）', 'ヨミ': 'ノケン', 'phon': ['n', 'o', 'k', 'e', 'N']}}

        self.tlpa3 = {
            0:{'orth': 'きいび', 'note': '（いびき）', 'ヨミ': 'キイビ', 'phon': ['k', 'i:', 'b', 'i']},
            1:{'orth': 'とびら', 'note': '', 'ヨミ': 'トビラ', 'phon': ['t', 'o', 'b', 'i', 'r', 'a']},
            2:{'orth': 'ろいり', 'note': '（いろり）', 'ヨミ': 'ロイリ', 'phon': ['r', 'o', 'i', 'r', 'i']},
            3:{'orth': 'つまげ', 'note': '（まつげ）', 'ヨミ': 'ツマゲ', 'phon': ['ts', 'u', 'm', 'a', 'g', 'e']},
            4:{'orth': 'あられ', 'note': '', 'ヨミ': 'アラレ', 'phon': ['a', 'r', 'a', 'r', 'e']},
            5:{'orth': 'かかし', 'note': '', 'ヨミ': 'カカシ', 'phon': ['k', 'a', 'k', 'a', 'sh', 'i']},
            6:{'orth': 'えくぼ', 'note': '', 'ヨミ': 'エクボ', 'phon': ['e', 'k', 'u', 'b', 'o']},
            7:{'orth': 'ういが', 'note': '（うがい）', 'ヨミ': 'ウイガ', 'phon': ['u', 'i', 'g', 'a']},
            8:{'orth': 'けじめ', 'note': '', 'ヨミ': 'ケジメ', 'phon': ['k', 'e', 'j', 'i', 'm', 'e']},
            9:{'orth': 'めがね', 'note': '', 'ヨミ': 'メガネ', 'phon': ['m', 'e', 'g', 'a', 'n', 'e']},
            10:{'orth': 'みれぞ', 'note': '（みぞれ', 'ヨミ': 'ミレゾ', 'phon': ['m', 'i', 'r', 'e', 'z', 'o']},
            11:{'orth': 'まけお', 'note': '（おまけ）', 'ヨミ': 'マケオ', 'phon': ['m', 'a', 'k', 'e', 'o']},
            12:{'orth': 'つぽみ', 'note': '', 'ヨミ': 'ツポミ', 'phon': ['ts', 'u', 'p', 'o', 'm', 'i']},
            13:{'orth': 'みげや', 'note': '（みやげ）', 'ヨミ': 'ミゲヤ', 'phon': ['m', 'i', 'g', 'e', 'y', 'a']},
            14:{'orth': 'ねがめ', 'note': '（めがね）', 'ヨミ': 'ネガメ', 'phon': ['n', 'e', 'g', 'a', 'm', 'e']},
            15:{'orth': 'ともた', 'note': '（たもと）', 'ヨミ': 'トモタ', 'phon': ['t', 'o', 'm', 'o', 't', 'a']},
            16:{'orth': 'うわさ', 'note': '', 'ヨミ': 'ウワサ', 'phon': ['u', 'w', 'a', 's', 'a']},
            17:{'orth': 'きかね', 'note': '（かきね）', 'ヨミ': 'キカネ', 'phon': ['k', 'i', 'k', 'a', 'n', 'e']},
            18:{'orth': 'つくえ', 'note': '', 'ヨミ': 'ツクエ', 'phon': ['ts', 'u', 'k', 'u', 'e']},
            19:{'orth': 'かおり', 'note': '', 'ヨミ': 'カオリ', 'phon': ['k', 'a', 'o', 'r', 'i']},
            20:{'orth': 'うがい', 'note': '', 'ヨミ': 'ウガイ', 'phon': ['u', 'g', 'a', 'i']},
            21:{'orth': 'みぞれ', 'note': '', 'ヨミ': 'ミゾレ', 'phon': ['m', 'i', 'z', 'o', 'r', 'e']},
            22:{'orth': 'りかお', 'note': '（かおり）', 'ヨミ': 'リカオ', 'phon': ['r', 'i', 'k', 'a', 'o']},
            23:{'orth': 'おまけ', 'note': '', 'ヨミ': 'オマケ', 'phon': ['o', 'm', 'a', 'k', 'e']},
            24:{'orth': 'かきね', 'note': '', 'ヨミ': 'カキネ', 'phon': ['k', 'a', 'k', 'i', 'n', 'e']},
            25:{'orth': 'いろり', 'note': '', 'ヨミ': 'イロリ', 'phon': ['i', 'r', 'o', 'r', 'i']},
            26:{'orth': 'びとら', 'note': '（とびら）', 'ヨミ': 'びとら', 'phon': ['b', 'i', 't', 'o', 'r', 'a']},
            27:{'orth': 'えつく', 'note': '（つくえ）', 'ヨミ': 'エツク', 'phon': ['e', 'ts', 'u', 'k', 'u']},
            28:{'orth': 'ぼえく', 'note': '（えくぼ）', 'ヨミ': 'ボエク', 'phon': ['b', 'o', 'e', 'k', 'u']},
            29:{'orth': 'まつげ', 'note': '', 'ヨミ': 'マツゲ', 'phon': ['m', 'a', 'ts', 'u', 'g', 'e']},
            30:{'orth': 'わさう', 'note': '（うわさ）', 'ヨミ': 'ワサウ', 'phon': ['w', 'a', 's', 'a', 'u']},
            31:{'orth': 'ぼみつ', 'note': '（つぼみ）', 'ヨミ': 'ボミツ', 'phon': ['b', 'o', 'm', 'i', 'ts', 'u']},
            32:{'orth': 'かしか', 'note': '（かかし', 'ヨミ': 'カシカ', 'phon': ['k', 'a', 'sh', 'i', 'k', 'a']},
            33:{'orth': 'いびき', 'note': '', 'ヨミ': 'イビキ', 'phon': ['i', 'b', 'i', 'k', 'i']},
            34:{'orth': 'れらあ', 'note': '（あられ）', 'ヨミ': 'レラア', 'phon': ['r', 'e', 'r', 'a:']},
            35:{'orth': 'もなか', 'note': '', 'ヨミ': 'モナカ', 'phon': ['m', 'o', 'n', 'a', 'k', 'a']},
            36:{'orth': 'たもと', 'note': '', 'ヨミ': 'タモト', 'phon': ['t', 'a', 'm', 'o', 't', 'o']},
            37:{'orth': 'けめじ', 'note': '（けじめ', 'ヨミ': 'ケメジ', 'phon': ['k', 'e', 'm', 'e', 'j', 'i']},
            38:{'orth': 'みやげ', 'note': '', 'ヨミ': 'ミヤゲ', 'phon': ['m', 'i', 'y', 'a', 'g', 'e']},
            39:{'orth': 'かなも', 'note': '（もなか）', 'ヨミ': 'カナモ', 'phon': ['k', 'a', 'n', 'a', 'm', 'o']}}

        self.tlpa4 = {
            0:{'orth': 'ぬばた', 'note': '（おわり）', 'ヨミ': 'ヌバタ', 'phon': ['n', 'u', 'b', 'a', 't', 'a']},
            1:{'orth': 'かけら', 'note': '', 'ヨミ': 'カケラ', 'phon': ['k', 'a', 'k', 'e', 'r', 'a']},
            2:{'orth': 'たちる', 'note': '（つまみ）', 'ヨミ': 'タチル', 'phon': ['t', 'a', 'ch', 'i', 'r', 'u']},
            3:{'orth': 'どびん', 'note': '', 'ヨミ': 'ドビン', 'phon': ['d', 'o', 'b', 'i', 'N']},
            4:{'orth': 'ねうろ', 'note': '（みさき）', 'ヨミ': 'ネウロ', 'phon': ['n', 'e', 'u', 'r', 'o']},
            5:{'orth': 'ねらい', 'note': '', 'ヨミ': 'ネライ', 'phon': ['n', 'e', 'r', 'a', 'i']},
            6:{'orth': 'たすき', 'note': '', 'ヨミ': 'タスキ', 'phon': ['t', 'a', 's', 'u', 'k', 'i']},
            7:{'orth': 'つまみ', 'note': '', 'ヨミ': 'ツマミ', 'phon': ['ts', 'u', 'm', 'a', 'm', 'i']},
            8:{'orth': 'くじゆ', 'note': '（かけら)', 'ヨミ': 'クジユ', 'phon': ['k', 'u', 'j', 'i', 'y', 'u']},
            9:{'orth': 'うかき', 'note': '（たばこ）', 'ヨミ': 'ウカキ', 'phon': ['u', 'k', 'a', 'k', 'i']},
            10:{'orth': 'おはぎ', 'note': '', 'ヨミ': 'オハギ', 'phon': ['o', 'h', 'a', 'g', 'i']},
            11:{'orth': 'おさげ', 'note': '', 'ヨミ': 'オサゲ', 'phon': ['o', 's', 'a', 'g', 'e']},
            12:{'orth': 'とのぐ', 'note': '（あせも）', 'ヨミ': 'トノぐ', 'phon': ['t', 'o', 'n', 'o', 'g', 'u']},
            13:{'orth': 'におい', 'note': '', 'ヨミ': 'ニオイ', 'phon': ['n', 'i', 'o', 'i']},
            14:{'orth': 'よあな', 'note': '（におい）', 'ヨミ': 'ヨアナ', 'phon': ['y', 'o', 'a', 'n', 'a']},
            15:{'orth': 'ちりえ', 'note': '（えがお)', 'ヨミ': 'チリエ', 'phon': ['ch', 'i', 'r', 'i', 'e']},
            16:{'orth': 'もやら', 'note': '（どびん）', 'ヨミ': 'モヤラ', 'phon': ['m', 'o', 'y', 'a', 'r', 'a']},
            17:{'orth': 'こずえ', 'note': '', 'ヨミ': 'コズエ', 'phon': ['k', 'o', 'z', 'u', 'e']},
            18:{'orth': 'おわり', 'note': '', 'ヨミ': 'オワリ', 'phon': ['o', 'w', 'a', 'r', 'i']},
            19:{'orth': 'とでわ', 'note': '（おさげ）', 'ヨミ': 'トデワ', 'phon': ['t', 'o', 'd', 'e', 'w', 'a']},
            20:{'orth': 'ぼにん', 'note': '（おどり）', 'ヨミ': 'ボニン', 'phon': ['b', 'o', 'n', 'i', 'N']},
            21:{'orth': 'むかし', 'note': '', 'ヨミ': 'ムカシ', 'phon': ['m', 'u', 'k', 'a', 'sh', 'i']},
            22:{'orth': 'われみ', 'note': '（むかし）', 'ヨミ': 'ワレミ', 'phon': ['w', 'a', 'r', 'e', 'm', 'i']},
            23:{'orth': 'まだら', 'note': '', 'ヨミ': 'マダラ', 'phon': ['m', 'a', 'd', 'a', 'r', 'a']},
            24:{'orth': 'みさき', 'note': '', 'ヨミ': 'ミサキ', 'phon': ['m', 'i', 's', 'a', 'k', 'i']},
            25:{'orth': 'おげる', 'note': '（ねらい）', 'ヨミ': 'オゲル', 'phon': ['o', 'g', 'e', 'r', 'u']},
            26:{'orth': 'てびお', 'note': '（まとめ）', 'ヨミ': 'テびお', 'phon': ['t', 'e', 'b', 'i', 'o']},
            27:{'orth': 'てらご', 'note': '（くぼみ）', 'ヨミ': 'テラゴ', 'phon': ['t', 'e', 'r', 'a', 'g', 'o']},
            28:{'orth': 'たばこ', 'note': '', 'ヨミ': 'タバコ', 'phon': ['t', 'a', 'b', 'a', 'k', 'o']},
            29:{'orth': 'まとめ', 'note': '', 'ヨミ': 'マトメ', 'phon': ['m', 'a', 't', 'o', 'm', 'e']},
            30:{'orth': 'きぬが', 'note': '（まだら）', 'ヨミ': 'キヌガ', 'phon': ['k', 'i', 'n', 'u', 'g', 'a']},
            31:{'orth': 'のんき', 'note': '', 'ヨミ': 'ノンキ', 'phon': ['n', 'o', 'N', 'k', 'i']},
            32:{'orth': 'あせも', 'note': '', 'ヨミ': 'アセモ', 'phon': ['a', 's', 'e', 'm', 'o']},
            33:{'orth': 'くぼみ', 'note': '', 'ヨミ': 'クボミ', 'phon': ['k', 'u', 'b', 'o', 'm', 'i']},
            34:{'orth': 'いねめ', 'note': '（こずえ）', 'ヨミ': 'イネメ', 'phon': ['i', 'n', 'e', 'm', 'e']},
            35:{'orth': 'おどり', 'note': '', 'ヨミ': 'オドリ', 'phon': ['o', 'd', 'o', 'r', 'i']},
            36:{'orth': 'くまな', 'note': '（おはぎ）', 'ヨミ': 'クマナ', 'phon': ['k', 'u', 'm', 'a', 'n', 'a']},
            37:{'orth': 'えがお', 'note': '', 'ヨミ': 'エガオ', 'phon': ['e', 'g', 'a', 'o']},
            38:{'orth': 'あけむ', 'note': '（たすき）', 'ヨミ': 'アケム', 'phon': ['a', 'k', 'e', 'm', 'u']},
            39:{'orth': 'みこれ', 'note': '（のんき）', 'ヨミ': 'ミコレ', 'phon': ['m', 'i', 'k', 'o', 'r', 'e']}}

        orth_maxlen, phon_maxlen = 3, 9
        self.orth_maxlen = 1 + 1
        self.phon_maxlen = phon_maxlen + 1

        if self.orth_maxlen > self.phon_maxlen:
            self.maxlen = orth_maxlen
        else:
            self.maxlen = self.phon_maxlen

        if self.task == 'tlpa1':
            self.data_dict = self.tlpa1
        elif self.task == 'tlpa2':
            self.data_dict = self.tlpa2
        elif self.task == 'tlpa3':
            self.data_dict = self.tlpa3
        elif self.task == 'tlpa4':
            self.data_dict = self.tlpa4
        else:
           raise 'Invalid task'

        orth2info_dict = {}
        for k, v in self.data_dict.items():
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



class SALA_Dataset(RAM_Dataset):
    def __init__(self,
                 source:str="orth",
                 target:str="phon",
                 task:str="sala_r29",   # ['sala_r29', 'sala_r30', 'sala_r31]
                 max_words:int=None):


        super().__init__()
        self.task = task
        self.phon_list = super().get_phon_list()
        self.jchar_list = super().get_jchar_list()
        self.joyo_chars = super().get_joyo_chars()
        self.source = source
        self.target = target
        self.orth_list = self.jchar_list

        self.r29 = {0: {'orth': 'てれび', 'phon': ['t', 'e', 'r', 'e', 'b', 'i'], 'cond': '1000'},
                    1: {'orth': 'とびら', 'phon': ['t', 'o', 'b', 'i', 'r', 'a'], 'cond': '0100'},
                    2: {'orth': 'ものみ', 'phon': ['m', 'o', 'n', 'o', 'm', 'i'], 'cond': '0001'},
                    3: {'orth': 'こゆび', 'phon': ['k', 'o', 'y', 'u', 'b', 'i'], 'cond': '0100'},
                    4: {'orth': 'しょくじ', 'phon': ['sh', 'o', 'k', 'u', 'j', 'i'], 'cond': '1000'},
                    5: {'orth': 'よぼう', 'phon': ['y', 'o', 'b', 'o:'], 'cond': '0010'},
                    6: {'orth': 'でんわ', 'phon': ['d', 'e', 'N', 'w', 'a'], 'cond': '1000'},
                    7: {'orth': 'ひはん', 'phon': ['h', 'i', 'h', 'a', 'N'], 'cond': '0010'},
                    8: {'orth': 'はだか', 'phon': ['h', 'a', 'd', 'a', 'k', 'a'], 'cond': '0100'},
                    9: {'orth': 'たいこ', 'phon': ['t', 'a', 'i', 'k', 'o'], 'cond': '0001'},
                    10: {'orth': 'つごう', 'phon': ['ts', 'u', 'g', 'o:'], 'cond': '0010'},
                    11: {'orth': 'うでわ', 'phon': ['u', 'd', 'e', 'w', 'a'], 'cond': '0100'},
                    12: {'orth': 'よてい', 'phon': ['y', 'o', 't', 'e', 'i'], 'cond': '0010'},
                    13: {'orth': 'めぼし', 'phon': ['m', 'e', 'b', 'o', 'sh', 'i'], 'cond': '0001'},
                    14: {'orth': 'ていど', 'phon': ['t', 'e', 'i', 'd', 'o'], 'cond': '0010'},
                    15: {'orth': 'くるま', 'phon': ['k', 'u', 'r', 'u', 'm', 'a'], 'cond': '1000'},
                    16: {'orth': 'ききめ', 'phon': ['k', 'i', 'k', 'i', 'm', 'e'], 'cond': '0001'},
                    17: {'orth': 'しゃしん', 'phon': ['sh', 'a', 'sh', 'i', 'N'], 'cond': '1000'},
                    18: {'orth': 'あひる', 'phon': ['a', 'h', 'i', 'r', 'u'], 'cond': '0100'},
                    19: {'orth': 'ぐもん', 'phon': ['g', 'u', 'm', 'o', 'N'], 'cond': '0001'},
                    20: {'orth': 'ろんり', 'phon': ['r', 'o', 'N', 'r', 'i'], 'cond': '0010'},
                    21: {'orth': 'わがし', 'phon': ['w', 'a', 'g', 'a', 'sh', 'i'], 'cond': '0100'},
                    22: {'orth': 'くすり', 'phon': ['k', 'u', 's', 'u', 'r', 'i'], 'cond': '1000'},
                    23: {'orth': 'ちゃしつ',  'phon': ['ch', 'a', 'sh', 'i', 'ts', 'u'],  'cond': '0100'},
                    24: {'orth': 'めやす', 'phon': ['m', 'e', 'y', 'a', 's', 'u'], 'cond': '0001'},
                    25: {'orth': 'からだ', 'phon': ['k', 'a', 'r', 'a', 'd', 'a'], 'cond': '1000'},
                    26: {'orth': 'さかき', 'phon': ['s', 'a', 'k', 'a', 'k', 'i'], 'cond': '0001'},
                    27: {'orth': 'きやく', 'phon': ['k', 'i', 'y', 'a', 'k', 'u'], 'cond': '0010'},
                    28: {'orth': 'ひでん', 'phon': ['h', 'i', 'd', 'e', 'N'], 'cond': '0001'},
                    29: {'orth': 'りゆう', 'phon': ['r', 'i', 'y', 'u:'], 'cond': '0010'},
                    30: {'orth': 'ぶらく', 'phon': ['b', 'u', 'r', 'a', 'k', 'u'], 'cond': '0001'},
                    31: {'orth': 'あぶら', 'phon': ['a', 'b', 'u', 'r', 'a'], 'cond': '1000'},
                    32: {'orth': 'ゆびわ', 'phon': ['y', 'u', 'b', 'i', 'w', 'a'], 'cond': '0100'},
                    33: {'orth': 'てがみ', 'phon': ['t', 'e', 'g', 'a', 'm', 'i'], 'cond': '1000'},
                    34: {'orth': 'とりい', 'phon': ['t', 'o', 'r', 'i:'], 'cond': '0100'},
                    35: {'orth': 'かしつ', 'phon': ['k', 'a', 'sh', 'i', 'ts', 'u'], 'cond': '0010'},
                    36: {'orth': 'いちば', 'phon': ['i', 'ch', 'i', 'b', 'a'], 'cond': '1000'},
                    37: {'orth': 'ゆみや', 'phon': ['y', 'u', 'm', 'i', 'y', 'a'], 'cond': '0100'},
                    38: {'orth': 'きみつ', 'phon': ['k', 'i', 'm', 'i', 'ts', 'u'], 'cond': '0010'},
                    39: {'orth': 'むめい', 'phon': ['m', 'u', 'm', 'e', 'i'], 'cond': '0001'},
                    40: {'orth': 'げんか', 'phon': ['g', 'e', 'N', 'k', 'a'], 'cond': '0010'},
                    41: {'orth': 'こども', 'phon': ['k', 'o', 'd', 'o', 'm', 'o'], 'cond': '1000'},
                    42: {'orth': 'びとく', 'phon': ['b', 'i', 't', 'o', 'k', 'u'], 'cond': '0001'},
                    43: {'orth': 'こんぶ', 'phon': ['k', 'o', 'N', 'b', 'u'], 'cond': '0100'},
                    44: {'orth': 'ひょうか', 'phon': ['hy', 'o:', 'k', 'a'], 'cond': '0010'},
                    45: {'orth': 'ひとみ', 'phon': ['h', 'i', 't', 'o', 'm', 'i'], 'cond': '0100'},
                    46: {'orth': 'ゆいしょ', 'phon': ['y', 'u', 'i', 'sh', 'o'], 'cond': '0001'},
                    47: {'orth': 'かざん', 'phon': ['k', 'a', 'z', 'a', 'N'], 'cond': '1000'},
                    48: {'orth': 'ひろま', 'phon': ['h', 'i', 'r', 'o', 'm', 'a'], 'cond': '0100'},
                    49: {'orth': 'あんじ', 'phon': ['a', 'N', 'j', 'i'], 'cond': '0010'},
                    50: {'orth': 'あとめ', 'phon': ['a', 't', 'o', 'm', 'e'], 'cond': '0001'},
                    51: {'orth': 'みどり', 'phon': ['m', 'i', 'd', 'o', 'r', 'i'], 'cond': '1000'}}

        self.r30 = {0: {'orth': 'パン', 'phon': ['p', 'a', 'N'], 'cond': ['kata', '2']},
                    1: {'orth': '信号', 'phon': ['sh', 'i', 'N', 'g', 'o:'], 'cond': ['kanj', '4']},
                    2: {'orth': '馬', 'phon': ['u', 'm', 'a'], 'cond': ['kanj', '2']},
                    3: {'orth': '紅茶', 'phon': ['k', 'o:', 'ch', 'a'], 'cond': ['kanj', '3']},
                    4: {'orth': 'えび', 'phon': ['e', 'b', 'i'], 'cond': ['hira', '2']},
                    5: {'orth': 'ポスト',  'phon': ['p', 'o', 's', 'u', 't', 'o'],  'cond': ['kata', '3']},
                    6: {'orth': 'にわとり', 'phon': ['n', 'i', 'w', 'a', 't', 'o', 'r', 'i'], 'cond': ['hira', '4']},
                    7: {'orth': '耳', 'phon': ['m', 'i', 'm', 'i'], 'cond': ['kanj', '2']},
                    8: {'orth': 'エプロン', 'phon': ['e', 'p', 'u', 'r', 'o', 'N'], 'cond': ['kata', '4']},
                    9: {'orth': '牛乳', 'phon': ['gy', 'u:', 'ny', 'u:'], 'cond': ['kanj', '4']},
                    10: {'orth': 'たわし', 'phon': ['t', 'a', 'w', 'a', 'sh', 'i'], 'cond': ['hira', '3']},
                    11: {'orth': 'じゃがいも', 'phon': ['j', 'a', 'g', 'a', 'i', 'm', 'o'], 'cond': ['hira', '4']},
                    12: {'orth': '切手', 'phon': ['k', 'i', 'q', 't', 'e'], 'cond': ['kanj', '3']},
                    13: {'orth': 'バス', 'phon': ['b', 'a', 's', 'u'], 'cond': ['kata', '2']},
                    14: {'orth': 'ちりとり', 'phon': ['ch', 'i', 'r', 'i', 't', 'o', 'r', 'i'], 'cond': ['hira', '4']},
                    15: {'orth': 'アルバム', 'phon': ['a', 'r', 'u', 'b', 'a', 'm', 'u'], 'cond': ['kata', '4']},
                    16: {'orth': 'あご', 'phon': ['a', 'g', 'o'], 'cond': ['hira', '2']},
                    17: {'orth': 'マスク', 'phon': ['m', 'a', 's', 'u', 'k', 'u'], 'cond': ['kata', '3']},
                    18: {'orth': '毛虫', 'phon': ['k', 'e', 'm', 'u', 'sh', 'i'], 'cond': ['kanj', '3']},
                    19: {'orth': 'りんご', 'phon': ['r', 'i', 'N', 'g', 'o'], 'cond': ['hira', '3']},
                    20: {'orth': '目薬',  'phon': ['m', 'e', 'g', 'u', 's', 'u', 'r', 'i'], 'cond': ['kanj', '4']},
                    21: {'orth': 'バナナ', 'phon': ['b', 'a', 'n', 'a', 'n', 'a'], 'cond': ['kata', '3']},
                    22: {'orth': 'トンネル', 'phon': ['t', 'o', 'N', 'n', 'e', 'r', 'u'], 'cond': ['kata', '4']},
                    23: {'orth': 'たこ', 'phon': ['t', 'a', 'k', 'o'], 'cond': ['hira', '2']},
                    24: {'orth': '太陽', 'phon': ['t', 'a', 'i', 'y', 'o:'], 'cond': ['kanj', '4']},
                    25: {'orth': 'ピザ', 'phon': ['p', 'i', 'z', 'a'], 'cond': ['kata', '2']},
                    26: {'orth': 'はさみ', 'phon': ['h', 'a', 's', 'a', 'm', 'i'], 'cond': ['hira', '3']},
                    27: {'orth': 'テレビ', 'phon': ['t', 'e', 'r', 'e', 'b', 'i'], 'cond': ['kata', '3']},
                    28: {'orth': 'あじさい', 'phon': ['a', 'j', 'i', 's', 'a', 'i'], 'cond': ['hira', '4']},
                    29: {'orth': 'こま', 'phon': ['k', 'o', 'm', 'a'], 'cond': ['hira', '2']},
                    30: {'orth': 'カーテン', 'phon': ['k', 'a:', 't', 'e', 'N'], 'cond': ['kata', '4']},
                    31: {'orth': 'ねぎ', 'phon': ['n', 'e', 'g', 'i'], 'cond': ['hira', '2']},
                    32: {'orth': '学校', 'phon': ['g', 'a', 'q', 'k', 'o:'], 'cond': ['kanj', '4']},
                    33: {'orth': 'タオル', 'phon': ['t', 'a', 'o', 'r', 'u'], 'cond': ['kata', '3']},
                    34: {'orth': '虹', 'phon': ['n', 'i', 'j', 'i'], 'cond': ['kanj', '2']},
                    35: {'orth': 'バラ', 'phon': ['b', 'a', 'r', 'a'], 'cond': ['kata', '2']},
                    36: {'orth': '水着', 'phon': ['m', 'i', 'z', 'u', 'g', 'i'], 'cond': ['kanj', '3']},
                    37: {'orth': 'うさぎ', 'phon': ['u', 's', 'a', 'g', 'i'], 'cond': ['hira', '3']},
                    38: {'orth': 'ダム', 'phon': ['d', 'a', 'm', 'u'], 'cond': ['kata', '2']},
                    39: {'orth': '空', 'phon': ['s', 'o', 'r', 'a'], 'cond': ['kanj', '2']},
                    40: {'orth': '電話', 'phon': ['d', 'e', 'N', 'w', 'a'], 'cond': ['kanj', '3']},
                    41: {'orth': 'ライオン', 'phon': ['r', 'a', 'i', 'o', 'N'], 'cond': ['kata', '4']},
                    42: {'orth': 'やかん', 'phon': ['y', 'a', 'k', 'a', 'N'], 'cond': ['hira', '3']},
                    43: {'orth': '指', 'phon': ['y', 'u', 'b', 'i'], 'cond': ['kanj', '2']},
                    44: {'orth': 'おにぎり', 'phon': ['o', 'n', 'i', 'g', 'i', 'r', 'i'], 'cond': ['hira', '4']},
                    45: {'orth': 'ドーナツ', 'phon': ['d', 'o:', 'n', 'a', 'ts', 'u'], 'cond': ['kata', '4']},
                    46: {'orth': '交番', 'phon': ['k', 'o:', 'b', 'a', 'N'], 'cond': ['kanj', '4']},
                    47: {'orth': 'いちご', 'phon': ['i', 'ch', 'i', 'g', 'o'], 'cond': ['hira', '3']},
                    48: {'orth': 'ドア', 'phon': ['d', 'o', 'a'], 'cond': ['kata', '2']},
                    49: {'orth': 'ろうそく', 'phon': ['r', 'o:', 's', 'o', 'k', 'u'], 'cond': ['hira', '4']},
                    50: {'orth': '写真', 'phon': ['sh', 'a', 'sh', 'i', 'N'], 'cond': ['kanj', '3']},
                    51: {'orth': 'アンテナ', 'phon': ['a', 'N', 't', 'e', 'n', 'a'], 'cond': ['kata', '4']},
                    52: {'orth': 'へそ', 'phon': ['h', 'e', 's', 'o'], 'cond': ['hira', '2']},
                    53: {'orth': 'のこぎり', 'phon': ['n', 'o', 'k', 'o', 'g', 'i', 'r', 'i'], 'cond': ['hira', '4']},
                    54: {'orth': 'コップ', 'phon': ['k', 'o', 'q', 'p', 'u'], 'cond': ['kata', '3']},
                    55: {'orth': '滝', 'phon': ['t', 'a', 'k', 'i'], 'cond': ['kanj', '2']},
                    56: {'orth': '手紙', 'phon': ['t', 'e', 'g', 'a', 'm', 'i'], 'cond': ['kanj', '3']},
                    57: {'orth': 'こたつ', 'phon': ['k', 'o', 't', 'a', 'ts', 'u'], 'cond': ['hira', '3']},
                    58: {'orth': '牛', 'phon': ['u', 'sh', 'i'], 'cond': ['kanj', '2']},
                    59: {'orth': 'ピアノ', 'phon': ['p', 'i', 'a', 'n', 'o'], 'cond': ['kata', '3']},
                    60: {'orth': 'デモ', 'phon': ['d', 'e', 'm', 'o'], 'cond': ['kata', '2']},
                    61: {'orth': '金魚', 'phon': ['k', 'i', 'N', 'gy', 'o'], 'cond': ['kanj', '3']},
                    62: {'orth': 'ペン', 'phon': ['p', 'e', 'N'], 'cond': ['kata', '2']},
                    63: {'orth': '弁当', 'phon': ['b', 'e', 'N', 't', 'o:'], 'cond': ['kanj', '4']},
                    64: {'orth': 'こけし', 'phon': ['k', 'o', 'k', 'e', 'sh', 'i'], 'cond': ['hira', '3']},
                    65: {'orth': '骨', 'phon': ['h', 'o', 'n', 'e'], 'cond': ['kanj', '2']},
                    66: {'orth': '朝顔', 'phon': ['a', 's', 'a', 'g', 'a', 'o'], 'cond': ['kanj', '4']},
                    67: {'orth': 'ミシン', 'phon': ['m', 'i', 'sh', 'i', 'N'], 'cond': ['kata', '3']},
                    68: {'orth': 'いか', 'phon': ['i', 'k', 'a'], 'cond': ['hira', '2']},
                    69: {'orth': 'アイロン', 'phon': ['a', 'i', 'r', 'o', 'N'], 'cond': ['kata', '4']},
                    70: {'orth': 'かまぼこ', 'phon': ['k', 'a', 'm', 'a', 'b', 'o', 'k', 'o'], 'cond': ['hira', '4']},
                    71: {'orth': 'バット', 'phon': ['b', 'a', 'q', 't', 'o'], 'cond': ['kata', '3']},
                    72: {'orth': 'やぎ', 'phon': ['y', 'a', 'g', 'i'], 'cond': ['hira', '2']},
                    73: {'orth': '火山', 'phon': ['k', 'a', 'z', 'a', 'N'], 'cond': ['kanj', '3']},
                    74: {'orth': 'ネクタイ', 'phon': ['n', 'e', 'k', 'u', 't', 'a', 'i'], 'cond': ['kata', '4']},
                    75: {'orth': '窓', 'phon': ['m', 'a', 'd', 'o'], 'cond': ['kanj', '2']},
                    76: {'orth': 'ペンギン', 'phon': ['p', 'e', 'N', 'g', 'i', 'N'], 'cond': ['kata', '4']},
                    77: {'orth': 'うちわ', 'phon': ['u', 'ch', 'i', 'w', 'a'], 'cond': ['hira', '3']},
                    78: {'orth': '灰皿', 'phon': ['h', 'a', 'i', 'z', 'a', 'r', 'a'], 'cond': ['kanj', '4']},
                    79: {'orth': 'ガム', 'phon': ['g', 'a', 'm', 'u'], 'cond': ['kata', '2']},
                    80: {'orth': 'ひまわり', 'phon': ['h', 'i', 'm', 'a', 'w', 'a', 'r', 'i'], 'cond': ['hira', '4']},
                    81: {'orth': 'キャベツ', 'phon': ['ky', 'a', 'b', 'e', 'ts', 'u'], 'cond': ['kata', '3']},
                    82: {'orth': '口', 'phon': ['k', 'u', 'ch', 'i'], 'cond': ['kanj', '2']},
                    83: {'orth': 'しゃもじ', 'phon': ['sh', 'a', 'm', 'o', 'j', 'i'], 'cond': ['hira', '3']},
                    84: {'orth': 'ふぐ', 'phon': ['f', 'u', 'g', 'u'], 'cond': ['hira', '2']},
                    85: {'orth': '背中', 'phon': ['s', 'e', 'n', 'a', 'k', 'a'], 'cond': ['kanj', '3']},
                    86: {'orth': 'なす', 'phon': ['n', 'a', 's', 'u'], 'cond': ['hira', '2']},
                    87: {'orth': 'そろばん', 'phon': ['s', 'o', 'r', 'o', 'b', 'a', 'N'], 'cond': ['hira', '4']},
                    88: {'orth': 'ハム', 'phon': ['h', 'a', 'm', 'u'], 'cond': ['kata', '2']},
                    89: {'orth': '病院', 'phon': ['by', 'o:', 'i', 'N'], 'cond': ['kanj', '4']}}

        self.r31 = {0: {'orth': 'のい', 'phon': ['n', 'o', 'i'], 'cond': '2'},
                    1: {'orth': 'びげ', 'phon': ['b', 'i', 'g', 'e'], 'cond': '2'},
                    2: {'orth': 'ずきょ', 'phon': ['z', 'u', 'ky', 'o'], 'cond': '2'},
                    3: {'orth': 'じず', 'phon': ['j', 'i', 'z', 'u'], 'cond': '2'},
                    4: {'orth': 'えも', 'phon': ['e', 'm', 'o'], 'cond': '2'},
                    5: {'orth': 'すお', 'phon': ['s', 'u', 'o'], 'cond': '2'},
                    6: {'orth': 'きゃし', 'phon': ['ky', 'a', 'sh', 'i'], 'cond': '2'},
                    7: {'orth': 'のゆ', 'phon': ['n', 'o', 'y', 'u'], 'cond': '2'},
                    8: {'orth': 'けぼ', 'phon': ['k', 'e', 'b', 'o'], 'cond': '2'},
                    9: {'orth': 'やび', 'phon': ['y', 'a', 'b', 'i'], 'cond': '2'},
                    10: {'orth': 'けぐ', 'phon': ['k', 'e', 'g', 'u'], 'cond': '2'},
                    11: {'orth': 'ばべ', 'phon': ['b', 'a', 'b', 'e'], 'cond': '2'},
                    12: {'orth': 'けひ', 'phon': ['k', 'e', 'h', 'i'], 'cond': '2'},
                    13: {'orth': 'ゆち', 'phon': ['y', 'u', 'ch', 'i'], 'cond': '2'},
                    14: {'orth': 'とぷか', 'phon': ['t', 'o', 'p', 'u', 'k', 'a'], 'cond': '3'},
                    15: {'orth': 'にょせき', 'phon': ['ny', 'o', 's', 'e', 'k', 'i'], 'cond': '3'},
                    16: {'orth': 'えなぶ', 'phon': ['e', 'n', 'a', 'b', 'u'], 'cond': '3'},
                    17: {'orth': 'かきふ', 'phon': ['k', 'a', 'k', 'i', 'f', 'u'], 'cond': '3'},
                    18: {'orth': 'こうな', 'phon': ['k', 'o:', 'n', 'a'], 'cond': '3'},
                    19: {'orth': 'ねっけ', 'phon': ['n', 'e', 'q', 'k', 'e'], 'cond': '3'},
                    20: {'orth': 'くらが', 'phon': ['k', 'u', 'r', 'a', 'g', 'a'], 'cond': '3'},
                    21: {'orth': 'ふけの', 'phon': ['f', 'u', 'k', 'e', 'n', 'o'], 'cond': '3'},
                    22: {'orth': 'めひゃく', 'phon': ['m', 'e', 'hy', 'a', 'k', 'u'], 'cond': '3'},
                    23: {'orth': 'いいと', 'phon': ['i:', 't', 'o'], 'cond': '3'},
                    24: {'orth': 'くれみゅ', 'phon': ['k', 'u', 'r', 'e', 'my', 'u'], 'cond': '3'},
                    25: {'orth': 'みった', 'phon': ['m', 'i', 'q', 't', 'a'], 'cond': '3'},
                    26: {'orth': 'すんき', 'phon': ['s', 'u', 'N', 'k', 'i'], 'cond': '3'},
                    27: {'orth': 'らげん', 'phon': ['r', 'a', 'g', 'e', 'N'], 'cond': '3'},
                    28: {'orth': 'いまそし', 'phon': ['i', 'm', 'a', 's', 'o', 'sh', 'i'], 'cond': '4'},
                    29: {'orth': 'りゃくしけ', 'phon': ['ry', 'a', 'k', 'u', 'sh', 'i', 'k', 'e'], 'cond': '4'},
                    30: {'orth': 'にっさき', 'phon': ['n', 'i', 'q', 's', 'a', 'k', 'i'], 'cond': '4'},
                    31: {'orth': 'ざっこり', 'phon': ['z', 'a', 'q', 'k', 'o', 'r', 'i'], 'cond': '4'},
                    32: {'orth': 'びかゆう', 'phon': ['b', 'i', 'k', 'a', 'y', 'u:'], 'cond': '4'},
                    33: {'orth': 'さねきり', 'phon': ['s', 'a', 'n', 'e', 'k', 'i', 'r', 'i'], 'cond': '4'},
                    34: {'orth': 'けだずり', 'phon': ['k', 'e', 'd', 'a', 'z', 'u', 'r', 'i'], 'cond': '4'},
                    35: {'orth': 'きっこく', 'phon': ['k', 'i', 'q', 'k', 'o', 'k', 'u'], 'cond': '4'},
                    36: {'orth': 'ぱころね', 'phon': ['p', 'a', 'k', 'o', 'r', 'o', 'n', 'e'], 'cond': '4'},
                    37: {'orth': 'いかたん', 'phon': ['i', 'k', 'a', 't', 'a', 'N'], 'cond': '4'},
                    38: {'orth': 'ねいたく', 'phon': ['n', 'e', 'i', 't', 'a', 'k', 'u'], 'cond': '4'},
                    39: {'orth': 'てずしな', 'phon': ['t', 'e', 'z', 'u', 'sh', 'i', 'n', 'a'], 'cond': '4'},
                    40: {'orth': 'ひごちゃに', 'phon': ['h', 'i', 'g', 'o', 'ch', 'a', 'n', 'i'], 'cond': '4'},
                    41: {'orth': 'ふばんず', 'phon': ['f', 'u', 'b', 'a', 'N', 'z', 'u'], 'cond': '4'},
                    42: {'orth': 'さかじょがた', 'phon': ['s', 'a', 'k', 'a', 'j', 'o', 'g', 'a', 't', 'a'], 'cond': '5'},
                    43: {'orth': 'まとびこう', 'phon': ['m', 'a', 't', 'o', 'b', 'i', 'k', 'o:'], 'cond': '5'},
                    44: {'orth': 'しごふうか', 'phon': ['sh', 'i', 'g', 'o', 'f', 'u:', 'k', 'a'], 'cond': '5'},
                    45: {'orth': 'かすじごと', 'phon': ['k', 'a', 's', 'u', 'j', 'i', 'g', 'o', 't', 'o'], 'cond': '5'},
                    46: {'orth': 'ごふれがん', 'phon': ['g', 'o', 'f', 'u', 'r', 'e', 'g', 'a', 'N'], 'cond': '5'},
                    47: {'orth': 'なかしりん', 'phon': ['n', 'a', 'k', 'a', 'sh', 'i', 'r', 'i', 'N'], 'cond': '5'},
                    48: {'orth': 'おんさつか', 'phon': ['o', 'N', 's', 'a', 'ts', 'u', 'k', 'a'], 'cond': '5'},
                    49: {'orth': 'きゅらぶっし', 'phon': ['ky', 'u', 'r', 'a', 'b', 'u', 'q', 'sh', 'i'], 'cond': '5'},
                    50: {'orth': 'しっつくみ', 'phon': ['sh', 'i', 'q', 'ts', 'u', 'k', 'u', 'm', 'i'], 'cond': '5'},
                    51: {'orth': 'ふせざわり', 'phon': ['f', 'u', 's', 'e', 'z', 'a', 'w', 'a', 'r', 'i'], 'cond': '5'},
                    52: {'orth': 'きなくさら', 'phon': ['k', 'i', 'n', 'a', 'k', 'u', 's', 'a', 'r', 'a'], 'cond': '5'},
                    53: {'orth': 'つきのとめ', 'phon': ['ts', 'u', 'k', 'i', 'n', 'o', 't', 'o', 'm', 'e'], 'cond': '5'},
                    54: {'orth': 'えなふりぎょ', 'phon': ['e', 'n', 'a', 'f', 'u', 'r', 'i', 'gy', 'o'], 'cond': '5'},
                    55: {'orth': 'しきっぱく', 'phon': ['sh', 'i', 'k', 'i', 'q', 'p', 'a', 'k', 'u'], 'cond': '5'}}

        if self.task == 'sala_r29':
            self.data_dict = self.r29
        elif self.task == 'sala_r30':
            self.data_dict = self.r30
        elif self.task == 'sala_r31':
            self.data_dict = self.r31
        else:
           raise 'Invalid task'

        orth_maxlen, phon_maxlen = 0, 0
        for k, v in self.data_dict.items():
            _len = len(v['orth'])
            if _len > orth_maxlen:
                orth_maxlen = _len
            _len = len(v['phon'])
            if _len > phon_maxlen:
                phon_maxlen = _len

        self.orth_maxlen = 1 + 1
        self.phon_maxlen = phon_maxlen + 1

        if self.orth_maxlen > self.phon_maxlen:
            self.maxlen = orth_maxlen
        else:
            self.maxlen = self.phon_maxlen

        orth2info_dict = {}
        for k, v in self.data_dict.items():
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
