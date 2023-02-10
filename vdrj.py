#cover_rate = 0.99  # カバー率の設定
import torch
import numpy as np
import requests
import pandas as pd
import os
import jaconv
import datetime

class VDRJ_dataset(torch.utils.data.Dataset):

    def __init__(self,
                 cover_rate:float=0.99,
                 verbose:bool=False,
                 make_dataframe:bool=True):

        """
        日本語を読むための語彙データベース（VDRJ） Ver. 1.1　（＝日本語を読むための”ＴＭ語彙リスト”（総合版）　Ver.4.0）
        を読み込んで，「単語」と対応する「読み」のリストを返す

        引数:
        cover_rate:float
        松下コーパスの `(Fw)累積テキストカバー率（想定既知語彙分を含む)` に基づいて語彙を選択するためのカバー率
        デフォルトでは 0.99 だが変更可能

        戻り値:
            X:dict 以下のキーを持つ辞書
            'lexeme': 語彙素
            'orth': 書記素
            'yomi': よみ
            'mora': よみのモーラ
            'cover_r': 累積カバー率
        df:Pandas.DataFrame
        上記の辞書を pandas.DataFrame にした実体
        """

        vdrj_url='http://www17408ui.sakura.ne.jp/tatsum/database/VDRJ_Ver1_1_Research_Top60894.xlsx'
        excel_fname = vdrj_url.split('/')[-1]  # 直上行の url からエクセルファイル名を切り出す

        # もしエクセルファイルが存在しなかったら ダウンロードする
        if not os.path.exists(excel_fname):
            print(f'エクセルファイルのダウンロード {datetime.datetime.now()}...') if verbose else None
            r = requests.get(vdrj_url)
            with open(excel_fname, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                print('Downloading {0} - {1} bytes'.format(excel_fname, (total_length)))
                f.write(r.content)
            print(f'done {datetime.datetime.now()}') if verbose else None

        # 実際のエクセルファイルの読み込み
        sheet_name='重要度順語彙リスト60894語'  # シート名を指定
        print(f'エクセルファイルの読み込み {datetime.datetime.now()}...') if verbose else None
        df = pd.read_excel(excel_fname, sheet_name=sheet_name)
        print(f'done. {datetime.datetime.now()}') if verbose else None

        print(f'データ作成 {datetime.datetime.now()}...') if verbose else None
        # 累積カバー率が `cover_rate` 以下の語を選択
        df = df[df['(Fw)累積テキストカバー率（想定既知語彙分を含む）\nFw Cumulative Text Coverage including Assumed Known Words']<cover_rate]

        # 品詞 POS が '名詞-普通名詞-一般' のみを抽出
        df = df[df['品詞\nPart of Speech']=='名詞-普通名詞-一般']

        df = df[['見出し語彙素\nLexeme',
                 '標準的（新聞）表記\nStandard (Newspaper) Orthography',
                 '標準的読み方（カタカナ）\nStandard Reading (Katakana)',
                 '品詞\nPart of Speech',
                 '(Fw)累積テキストカバー率（想定既知語彙分を含む）\nFw Cumulative Text Coverage including Assumed Known Words',
                 'ID',
                ]].dropna()
        # .dropna() により NaN を含む行を削除


        # 必要となる情報のみをリスト化: ['語彙素', '書記素', 'よみ', '累積カバー率', 'ID']
        Lexeme = df['見出し語彙素\nLexeme'].to_list()
        CoverR = df['(Fw)累積テキストカバー率（想定既知語彙分を含む）\nFw Cumulative Text Coverage including Assumed Known Words'].to_list()
        ID     = df['ID'].to_list()
        Ortho  = df['標準的（新聞）表記\nStandard (Newspaper) Orthography'].to_list()
        Yomi   = df['標準的読み方（カタカナ）\nStandard Reading (Katakana)'].to_list()

        # 結果を Python の辞書にまとめる
        X = {}
        fish3words = []
        for l, o, y, r, idx in zip(Lexeme, Ortho, Yomi, CoverR, ID):

            # `よみ` に '/' が含まれる項目 `ALT` を `オルト/エイエルティー` の場合最初のエントリだけを採用する
            if isinstance(y, str):
                y = jaconv.normalize(y)
                if '/' in y:
                    y = y.split('/')[0]
                    print(f'{y}, idx:{idx}')

            # `語彙素` に '鱻' が含まれる項目は '鱻' 以降を切り捨てる
            if isinstance(l, str):
                if '鱻' in l:
                    fish3words.append({'idx':idx, 'lexeme':l})
                    l = l.split('鱻')[0]

            if isinstance(l, str):
                lexeme = jaconv.normalize(l)  # 書記素を UTF-8 NKCD に正規化
            else:
                continue

            if isinstance(o, str):
                orth = jaconv.normalize(o)

                # 書記素が '*' になっている場合があるので，語彙素情報で置き換える
                orth = lexeme if orth == '*' else orth
            else:
                continue

            if isinstance(y, str):
                yomi = jaconv.hira2kata(y)
                hira = jaconv.kata2hira(yomi)
                phon = jaconv.hiragana2julius(hira).split(' ')
            else:
                continue

            X[idx] = {'lexeme': lexeme,
                      'orth': orth,
                      'yomi': yomi,
                      'phon': phon,
                      'cover_r': r}

        #print(fish3words) if verbose else None
        self.df = pd.DataFrame.from_dict(X, orient='index')
        self.X = X
        print(f'done {datetime.datetime.now()}...') if verbose else None
        print(f'カバー率:{cover_rate*100:.2f} % により，単語数:{len(X)} 語を抽出') if verbose else None

        self.phon_vocab = ['<PAD>', '<EOW>', '<SOW>', '<UNK>',
                           'N', 'a', 'a:', 'e', 'e:', 'i', 'i:', 'i::', 'o', 'o:', 'o::', 'u', 'u:',
                           'b', 'by', 'ch', 'd', 'dy', 'f', 'g', 'gy', 'h', 'hy', 'j', 'k', 'ky',
                           'm', 'my', 'n', 'ny', 'p', 'py', 'q', 'r', 'ry', 's', 'sh', 't', 'ts', 'w', 'y', 'z']



	def __len__(self):
		return len(self.X)

	def __getitem(self, idx)


    def set_source_and_target_from_params(
            self,
            source:str='orth',
            target:str='phon',
            is_print:bool=True):

        # ソースとターゲットを設定しておく

        if source == 'orth':
            self.source_vocab = self.orth_vocab
            self.source_ids = 'orth_ids'
            self.source_maxlen = self.max_orth_length
            self.source_ids2tkn = self.orth_ids2tkn
            self.source_tkn2ids = self.orth_tkn2ids
        elif source == 'phon':
            self.source_vocab = self.phon_vocab
            self.source_ids = 'phon_ids'
            self.source_maxlen = self.max_phon_length
            self.source_ids2tkn = self.phon_ids2tkn
            self.source_tkn2ids = self.phon_tkn2ids
        elif source == 'mora':
            self.source_vocab = self.mora_vocab
            self.source_ids = 'mora_ids'
            self.source_maxlen = self.max_mora_length
            #self.source_ids2tkn = self.mora_ids2tkn
            #self.source_tkn2ids = self.mora_tkn2ids
        elif source == 'mora_p':
            self.source_vocab = self.mora_p_vocab
            self.source_ids = 'mora_p_ids'
            self.source_maxlen = self.max_mora_p_length
            #self.source_ids2tkn = self.mora_p_ids2tkn
            #self.source_tkn2ids = self.mora_p_tkn2ids
        elif source == 'mora_p_r':
            self.source_vocab = self.mora_p_vocab
            self.source_ids = 'mora_p_ids_r'
            self.source_maxlen = self.max_mora_p_length
            #self.source_ids2tkn = self.mora_p_r_ids2tkn
            #self.source_tkn2ids = self.mora_p_r_tkn2ids

        if target == 'orth':
            self.target_vocab = self.orth_vocab
            self.target_ids = 'orth_ids'
            self.target_maxlen = self.max_orth_length
            self.target_ids2tkn = self.orth_ids2tkn
            self.target_tkn2ids = self.orth_tkn2ids
        elif target == 'phon':
            self.target_vocab = self.phon_vocab
            self.target_ids = 'phon_ids'
            self.target_maxlen = self.max_phon_length
            self.target_ids2tkn = self.phon_ids2tkn
            self.target_tkn2ids = self.phon_tkn2ids
        elif target == 'mora':
            self.target_vocab = self.mora_vocab
            self.target_ids = 'mora_ids'
            self.target_maxlen = self.max_mora_length
            #self.target_ids2tkn = self.mora_ids2tkn
            #self.target_tkn2ids = self.mora_tkn2ids
        elif target == 'mora_p':
            self.target_vocab = self.mora_p_vocab
            self.target_ids = 'mora_p_ids'
            self.target_maxlen = self.max_mora_p_length
            #self.target_ids2tkn = self.mora_p_ids2tkn
            #self.target_tkn2ids = self.mora_p_tkn2ids
        elif target == 'mora_p_r':
            self.target_vocab = self.mora_p_vocab
            self.target_ids = 'mora_p_ids_r'
            self.target_maxlen = self.max_mora_p_length
            #self.target_ids2tkn = self.mora_p_r_ids2tkn
            #self.target_tkn2ids = self.mora_p__r_tkn2ids

        if is_print:
            print(colored(f'self.source:{self.source}', 'blue',
                          attrs=['bold']), f'{self.source_vocab}')
            print(colored(f'self.target:{target}', 'cyan',
                          attrs=['bold']), f'{self.target_vocab}')
            print(colored(f'self.source_ids:{self.source_ids}',
                          'blue', attrs=['bold']), f'{self.source_ids}')
            print(colored(f'self.target_ids:{self.target_ids}',
                          'cyan', attrs=['bold']), f'{self.target_ids}')

        #return  # source_vocab, source_ids, target_vocab, target_ids


