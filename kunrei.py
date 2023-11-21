from __future__ import unicode_literals
from copy import copy
import re
import unicodedata
"""
カタカナ表記されたヨミを，疑似的に音素に変換する。

訓令式については，内閣府のページを参照のこと

[現代仮名遣い　本文　第1（原則に基づくきまり）](https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/naikaku/gendaikana/honbun_dai1.html)
語を書き表すのに,現代語の音韻に従って,次の仮名を用いる。
ただし,下線を施した仮名は,第2に示す場合にだけ用いるものである。

1. 直音<br/>
<img src="https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/naikaku/img/1000004198_50on1.gif" style="width:256pt">

2. 拗音<br/>
<img src="https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/naikaku/img/1000005394_50on2.gif" style="width:256pt">

3.　撥音 `ん`<br/>
* 例: まなんで（学）　みなさん　しんねん（新年）　しゅんぶん（春分）

4.　促音 `っ`<br/>
* 例　　はしって（走）　かっき（活気）　がっこう（学校）　せっけん（石鹸＊）<br/>
　　〔注意〕促音に用いる「つ」は,なるべく小書きにする。

5.　長音

    1.　ア列の長音
         ア列の仮名に「あ」を添える。
        例　おかあさん　おばあさん
    2.　イ列の長音
         イ列の仮名に「い」を添える。
        例　にいさん　おじいさん
    3.　ウ列の長音
         ウ列の仮名に「う」を添える。
        例　おさむうございます（寒）　くうき（空気）　ふうふ（夫婦）
        うれしゅう存じます　きゅうり　ぼくじゅう（墨汁）　ちゅうもん（注文）
    4.　エ列の長音
        エ列の仮名に「え」を添える。
        例　ねえさん　ええ（応答の語）
    5.　オ列の長音
        オ列の仮名に｢う｣を添える。
        例　おとうさん　とうだい（灯台）
          わこうど（若人）　おうむ
          かおう（買）　あそぼう（遊）　おはよう（早）
          おうぎ（扇）　ほうる（放）　とう（塔）
          よいでしょう　はっぴょう（発表）
          きょう（今日）　ちょうちょう（蝶＊々）

HOME > 国語施策・日本語教育 > 国語施策情報 > 内閣告示・内閣訓令 > ローマ字のつづり方 > [ローマ字のつづり方　第1表・第2表](https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/naikaku/roma/honbun.html)

<img src="https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/naikaku/img/1000004856_kk_p194a.gif" style="width:384pt">
<img src="https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/naikaku/img/1000005492_111.gif" style="width:200pt">

[ローマ字のつづり方　解説](https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/naikaku/roma/kaisetu.html)

これは，昭和28年3月12日，国語審議会会長から文部大臣に建議した「ローマ字の単一化について」を政府として採択し，昭和 29 年 12 月 9 日に内閣告示第1号をもって告示したものです。
政府は，内閣告示と同じ日に内閣訓令第 1 号「ローマ字のつづり方の実施について」を発し，今後，各官庁はローマ字で国語を書き表す場合には，このつづり方によるべきことなどを訓令しました。

「ローマ字のつづり方」は，「まえがき，第1表，第2表，そえがき」から成っています。
第 1 表にはいわゆる訓令式のつづり方，第 2 表にはヘボン式（表の上から 5 列目まで）と日本式（6 列目以下）のつづり方のうち訓令式と異なるものだけを掲げてあり，一般に国語を書き表す場合は第1表に掲げたつづり方により，
国際的関係その他従来の慣例をにわかに改めがたい事情にある場合に限り，第 2 表によっても差し支えない旨を「まえがき」で述べています。

[ローマ字のつづり方　そえがき](ローマ字のつづり方　そえがき)

前表に定めたもののほか、おおむね次の各項による。

1. はねる音「ン」はすべてｎと書く。
2. はねる音を表わすｎと次にくる母音字またはｙとを切り離す必要がある場合には、ｎの次にアポストロフィを入れる。
3. つまる音は、最初の子音字を重ねて表わす。
4. 長音は母音字の上にアクサンシルコンフレックスをつけて表わす。なお、大文字の場合は、母音字を並べてもよい。
5. 特殊音の書き表わし方は自由とする
6. 文の書きはじめ、および固有名詞は語頭を大文字で書く。なお、固有名詞以外の名詞の語頭を大文字で書いてもよい。
"""

valid_katas = 'ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶー〜'


def kunrei(kata_str:str,               # 入力カタカナ文字列
              #_symbols:str=_symbols,   # 発音に関与しない記号
              is_ret_str:bool=False,   # 記号を削除した文字列を返すか否かの flag
              ):
    """Convert Katakana to kurei romaji.

    Parameters
    ----------
    kata_str : str
        Katakana string.

    Return
    ------
    str
        Alphabet (romaji) string.

    Examples
    --------
    >>> print(kunrei('テンキスゴクイイイイイイ'))
    t e N k i s u g o k u i:
    """

    reg_exp = f"r'[^{valid_katas}]'"
    out_str = re.sub(eval(reg_exp), '', kata_str)

    # 3 文字以上からなる変換規則
    # decomposed format UTF encodings されている `ウ゛` とそれに続く `ァィゥェォ` だと 3 文字になるので，
    out_str = out_str.replace('ウ゛ァ', ' b a')
    out_str = out_str.replace('ウ゛ィ', ' b i')
    out_str = out_str.replace('ウ゛ェ', ' b e')
    out_str = out_str.replace('ウ゛ォ', ' b o')
    out_str = out_str.replace('ウ゛ュ', ' by u')

    # composed 表現 `ゔ` + `[ぁぃぅぇぉ]` を追加
    out_str = out_str.replace('ヴァ', ' b a')
    out_str = out_str.replace('ヴィ', ' b i')
    out_str = out_str.replace('ヴェ', ' b e')
    out_str = out_str.replace('ヴォ', ' b o')
    out_str = out_str.replace('ヴュ', ' by u')

    # 2 文字からなる変換規則
    out_str = out_str.replace('ゥ゛', ' b u')

    out_str = out_str.replace('アァ', ' a a')
    out_str = out_str.replace('イィ', ' i i')
    out_str = out_str.replace('イェ', ' i e')
    out_str = out_str.replace('イャ', ' y a')
    out_str = out_str.replace('ウゥ', ' u:')
    out_str = out_str.replace('エェ', ' e e')
    out_str = out_str.replace('オォ', ' o:')

    out_str = out_str.replace('カァ', ' k a:')
    out_str = out_str.replace('キィ', ' k i:')
    out_str = out_str.replace('クゥ', ' k u:')
    out_str = out_str.replace('クャ', ' ky a')
    out_str = out_str.replace('クュ', ' ky u')
    out_str = out_str.replace('クョ', ' ky o')
    out_str = out_str.replace('ケェ', ' k e:')
    out_str = out_str.replace('コォ', ' k o:')
    out_str = out_str.replace('ガァ', ' g a:')
    out_str = out_str.replace('ギィ', ' g i:')
    out_str = out_str.replace('グゥ', ' g u:')
    out_str = out_str.replace('グャ', ' gy a')
    out_str = out_str.replace('グュ', ' gy u')
    out_str = out_str.replace('グョ', ' gy o')
    out_str = out_str.replace('ゲェ', ' g e:')
    out_str = out_str.replace('ゴォ', ' g o:')

    out_str = out_str.replace('サァ', ' s a:')
    out_str = out_str.replace('シィ', ' s i:')
    out_str = out_str.replace('スゥ', ' s u:')
    out_str = out_str.replace('スャ', ' s a')
    out_str = out_str.replace('スュ', ' s u')
    out_str = out_str.replace('スョ', ' s o')
    out_str = out_str.replace('セェ', ' s e:')
    out_str = out_str.replace('ソォ', ' s o:')
    out_str = out_str.replace('ザァ', ' z a:')
    out_str = out_str.replace('ジィ', ' j i:')
    out_str = out_str.replace('ズゥ', ' zy u:')
    out_str = out_str.replace('ズャ', ' zy a')
    out_str = out_str.replace('ズュ', ' zy u')
    out_str = out_str.replace('ズョ', ' zy o')
    out_str = out_str.replace('ゼェ', ' z e:')
    out_str = out_str.replace('ゾォ', ' z o:')
    out_str = out_str.replace('タァ', ' t a:')
    out_str = out_str.replace('チィ', ' ty i')
    out_str = out_str.replace('ツァ', ' ty a')
    out_str = out_str.replace('ツィ', ' ty i')
    out_str = out_str.replace('ツゥ', ' ty u')
    out_str = out_str.replace('ツャ', ' ty a')
    out_str = out_str.replace('ツュ', ' ty u')
    out_str = out_str.replace('ツョ', ' ty o')
    out_str = out_str.replace('ツェ', ' ty e')
    out_str = out_str.replace('ツォ', ' ty o')
    out_str = out_str.replace('テェ', ' ty e')
    out_str = out_str.replace('トォ', ' t o:')
    out_str = out_str.replace('ダァ', ' d a:')
    out_str = out_str.replace('ヂィ', ' d i:')
    out_str = out_str.replace('ヅゥ', ' d u:')
    out_str = out_str.replace('ヅャ', ' zy a')
    out_str = out_str.replace('ヅュ', ' zy u')
    out_str = out_str.replace('ヅョ', ' zy o')
    out_str = out_str.replace('デェ', ' d e:')
    out_str = out_str.replace('ドォ', ' d o:')
    out_str = out_str.replace('ナァ', ' n a:')
    out_str = out_str.replace('ニィ', ' n i:')
    out_str = out_str.replace('ヌゥ', ' n u:')
    out_str = out_str.replace('ヌャ', ' ny a')
    out_str = out_str.replace('ヌュ', ' ny u')
    out_str = out_str.replace('ヌョ', ' ny o')
    out_str = out_str.replace('ネェ', ' n e:')
    out_str = out_str.replace('ノォ', ' n o:')
    out_str = out_str.replace('ハァ', ' h a:')
    out_str = out_str.replace('ヒィ', ' h i:')
    out_str = out_str.replace('フゥ', ' f u:')
    out_str = out_str.replace('フャ', ' f a')
    out_str = out_str.replace('フュ', ' f u')
    out_str = out_str.replace('フョ', ' f o')
    out_str = out_str.replace('ヘェ', ' h e:')
    out_str = out_str.replace('ホォ', ' h o:')
    out_str = out_str.replace('バァ', ' b a:')
    out_str = out_str.replace('ビィ', ' b i:')
    out_str = out_str.replace('ブゥ', ' b u:')

    out_str = out_str.replace('ブュ', ' by u')
    out_str = out_str.replace('フョ', ' hy o')
    out_str = out_str.replace('ベェ', ' b e:')
    out_str = out_str.replace('ボォ', ' b o:')
    out_str = out_str.replace('パァ', ' p a:')
    out_str = out_str.replace('ピィ', ' p i:')
    out_str = out_str.replace('プゥ', ' p u:')
    out_str = out_str.replace('プャ', ' py a')
    out_str = out_str.replace('プュ', ' py u')
    out_str = out_str.replace('プョ', ' py o')
    out_str = out_str.replace('ペェ', ' p e:')
    out_str = out_str.replace('ポォ', ' p o:')
    out_str = out_str.replace('マァ', ' m a:')
    out_str = out_str.replace('ミィ', ' m i:')
    out_str = out_str.replace('ムゥ', ' m u:')
    out_str = out_str.replace('ムャ', ' my a')
    out_str = out_str.replace('ムュ', ' my u')
    out_str = out_str.replace('ムョ', ' my o')
    out_str = out_str.replace('メェ', ' m e:')
    out_str = out_str.replace('モォ', ' m o:')
    out_str = out_str.replace('ヤァ', ' y a:')
    out_str = out_str.replace('ユゥ', ' y u:')
    out_str = out_str.replace('ユャ', ' y a:')
    out_str = out_str.replace('ユュ', ' y u:')
    out_str = out_str.replace('ユョ', ' y o:')
    out_str = out_str.replace('ヨォ', ' y o:')
    out_str = out_str.replace('ラァ', ' r a:')
    out_str = out_str.replace('リィ', ' r i:')
    out_str = out_str.replace('ルゥ', ' r u:')
    out_str = out_str.replace('ルャ', ' ry a')
    out_str = out_str.replace('ルュ', ' ry u')
    out_str = out_str.replace('ルョ', ' ry o')
    out_str = out_str.replace('レェ', ' r e:')
    out_str = out_str.replace('ロォ', ' r o:')
    out_str = out_str.replace('ワァ', ' w a:')
    out_str = out_str.replace('ヲォ', ' o:')

    out_str = out_str.replace('ディ', ' d i')
    out_str = out_str.replace('デェ', ' d e:')
    out_str = out_str.replace('デャ', ' dy a')
    out_str = out_str.replace('デュ', ' dy u')
    out_str = out_str.replace('デョ', ' dy o')
    out_str = out_str.replace('ティ', ' t i')
    out_str = out_str.replace('テェ', ' t e:')
    out_str = out_str.replace('テャ', ' ty a')
    out_str = out_str.replace('テュ', ' ty u')
    out_str = out_str.replace('テョ', ' ty o')
    out_str = out_str.replace('スィ', ' s i')
    out_str = out_str.replace('ズァ', ' z u a')
    out_str = out_str.replace('ズィ', ' z i')
    out_str = out_str.replace('ズゥ', ' z u')
    out_str = out_str.replace('ズャ', ' zy a')
    out_str = out_str.replace('ズュ', ' zy u')
    out_str = out_str.replace('ズョ', ' zy o')
    out_str = out_str.replace('ズェ', ' z e')
    out_str = out_str.replace('ズォ', ' z o')

    out_str = out_str.replace('キャ', ' ky a')
    out_str = out_str.replace('キィ', ' ky i')
    out_str = out_str.replace('キュ', ' ky u')
    out_str = out_str.replace('キェ', ' ky e')
    out_str = out_str.replace('キョ', ' ky o')
    out_str = out_str.replace('キォ', ' ky o')
    out_str = out_str.replace('シャ', ' sy a')
    out_str = out_str.replace('シィ', ' sy i')
    out_str = out_str.replace('シュ', ' sy u')
    out_str = out_str.replace('シェ', ' sy e')
    out_str = out_str.replace('ショ', ' sy o')

    out_str = out_str.replace('タャ', ' ty a')
    out_str = out_str.replace('タィ', ' ty i')
    out_str = out_str.replace('タュ', ' ty u')
    out_str = out_str.replace('タェ', ' ty e')
    out_str = out_str.replace('タョ', ' ty o')

    out_str = out_str.replace('チャ', ' ch a')
    out_str = out_str.replace('チュ', ' ch u')
    out_str = out_str.replace('チェ', ' ch e')
    out_str = out_str.replace('チョ', ' ch o')

    out_str = out_str.replace('トゥ', ' t u')
    out_str = out_str.replace('トャ', ' ty a')
    out_str = out_str.replace('トュ', ' ty u')
    out_str = out_str.replace('トョ', ' ty o')
    out_str = out_str.replace('ドァ', ' d o a')
    out_str = out_str.replace('ドゥ', ' d u')
    out_str = out_str.replace('ドャ', ' dy a')
    out_str = out_str.replace('ドュ', ' dy u')
    out_str = out_str.replace('ドョ', ' dy o')
    out_str = out_str.replace('ドォ', ' d o:')
    out_str = out_str.replace('ニャ', ' ny a')
    out_str = out_str.replace('ニュ', ' ny u')
    out_str = out_str.replace('ニョ', ' ny o')
    out_str = out_str.replace('ヒャ', ' hy a')
    out_str = out_str.replace('ヒュ', ' hy u')
    out_str = out_str.replace('ヒョ', ' hy o')
    out_str = out_str.replace('ミャ', ' my a')
    out_str = out_str.replace('ミュ', ' my u')
    out_str = out_str.replace('ミョ', ' my o')
    out_str = out_str.replace('リャ', ' ry a')
    out_str = out_str.replace('リュ', ' ry u')
    out_str = out_str.replace('リョ', ' ry o')
    out_str = out_str.replace('ギャ', ' gy a')
    out_str = out_str.replace('ギュ', ' gy u')
    out_str = out_str.replace('ギョ', ' gy o')
    out_str = out_str.replace('ヂェ', ' j e')
    out_str = out_str.replace('ヂャ', ' j a')
    out_str = out_str.replace('ヂュ', ' j u')
    out_str = out_str.replace('ヂョ', ' j o')
    out_str = out_str.replace('ジェ', ' j e')
    out_str = out_str.replace('ジャ', ' j a')
    out_str = out_str.replace('ジュ', ' j u')
    out_str = out_str.replace('ジョ', ' j o')
    out_str = out_str.replace('ビャ', ' by a')
    out_str = out_str.replace('ビュ', ' by u')
    out_str = out_str.replace('ビョ', ' by o')
    out_str = out_str.replace('ピャ', ' py a')
    out_str = out_str.replace('ピュ', ' py u')
    out_str = out_str.replace('ピョ', ' py o')
    out_str = out_str.replace('ウァ', ' u a')
    out_str = out_str.replace('ウィ', ' w i')
    out_str = out_str.replace('ウェ', ' w e')
    out_str = out_str.replace('ウォ', ' w o')
    out_str = out_str.replace('ファ', ' f a')
    out_str = out_str.replace('フィ', ' f i')
    out_str = out_str.replace('フゥ', ' f u')
    out_str = out_str.replace('フャ', ' f a')
    out_str = out_str.replace('フュ', ' f u')
    out_str = out_str.replace('フョ', ' f o')
    out_str = out_str.replace('フェ', ' f e')
    out_str = out_str.replace('フォ', ' f o')

    # 1 音からなる変換規則
    out_str = out_str.replace('ウ゛', ' b u')

    out_str = out_str.replace('ア', ' a')
    out_str = out_str.replace('イ', ' i')
    out_str = out_str.replace('ウ', ' u')
    out_str = out_str.replace('エ', ' e')
    out_str = out_str.replace('オ', ' o')
    out_str = out_str.replace('カ', ' k a')
    out_str = out_str.replace('キ', ' k i')
    out_str = out_str.replace('ク', ' k u')
    out_str = out_str.replace('ケ', ' k e')
    out_str = out_str.replace('コ', ' k o')
    out_str = out_str.replace('サ', ' s a')
    out_str = out_str.replace('シ', ' s i')
    out_str = out_str.replace('ス', ' s u')
    out_str = out_str.replace('セ', ' s e')
    out_str = out_str.replace('ソ', ' s o')
    out_str = out_str.replace('タ', ' t a')
    out_str = out_str.replace('チ', ' t i')
    out_str = out_str.replace('ツ', ' t u')
    out_str = out_str.replace('テ', ' t e')
    out_str = out_str.replace('ト', ' t o')
    out_str = out_str.replace('ナ', ' n a')
    out_str = out_str.replace('ニ', ' n i')
    out_str = out_str.replace('ヌ', ' n u')
    out_str = out_str.replace('ネ', ' n e')
    out_str = out_str.replace('ノ', ' n o')
    out_str = out_str.replace('ハ', ' h a')
    out_str = out_str.replace('ヒ', ' h i')
    out_str = out_str.replace('フ', ' f u')
    out_str = out_str.replace('ヘ', ' h e')
    out_str = out_str.replace('ホ', ' h o')
    out_str = out_str.replace('マ', ' m a')
    out_str = out_str.replace('ミ', ' m i')
    out_str = out_str.replace('ム', ' m u')
    out_str = out_str.replace('メ', ' m e')
    out_str = out_str.replace('モ', ' m o')
    out_str = out_str.replace('ラ', ' r a')
    out_str = out_str.replace('リ', ' r i')
    out_str = out_str.replace('ル', ' r u')
    out_str = out_str.replace('レ', ' r e')
    out_str = out_str.replace('ロ', ' r o')
    out_str = out_str.replace('ガ', ' g a')
    out_str = out_str.replace('ギ', ' g i')
    out_str = out_str.replace('グ', ' g u')
    out_str = out_str.replace('ゲ', ' g e')
    out_str = out_str.replace('ゴ', ' g o')
    out_str = out_str.replace('ザ', ' z a')
    out_str = out_str.replace('ジ', ' j i')
    out_str = out_str.replace('ズ', ' z u')
    out_str = out_str.replace('ゼ', ' z e')
    out_str = out_str.replace('ゾ', ' z o')
    out_str = out_str.replace('ダ', ' d a')
    out_str = out_str.replace('ヂ', ' j i')
    out_str = out_str.replace('ヅ', ' z u')
    out_str = out_str.replace('デ', ' d e')
    out_str = out_str.replace('ド', ' d o')
    out_str = out_str.replace('バ', ' b a')
    out_str = out_str.replace('ビ', ' b i')
    out_str = out_str.replace('ブ', ' b u')
    out_str = out_str.replace('ベ', ' b e')
    out_str = out_str.replace('ボ', ' b o')
    out_str = out_str.replace('パ', ' p a')
    out_str = out_str.replace('ピ', ' p i')
    out_str = out_str.replace('プ', ' p u')
    out_str = out_str.replace('ペ', ' p e')
    out_str = out_str.replace('ポ', ' p o')
    out_str = out_str.replace('ヤ', ' y a')
    out_str = out_str.replace('ユ', ' y u')
    out_str = out_str.replace('ヨ', ' y o')
    out_str = out_str.replace('ワ', ' w a')
    out_str = out_str.replace('ヰ', ' i')
    out_str = out_str.replace('ヱ', ' e')
    out_str = out_str.replace('ン', ' N')
    out_str = out_str.replace('ッ', ' Q')
    # out_str = out_str.replace('ッ', ' q')
    # ここまでに処理されてない ァィゥェォ はそのまま大文字扱い
    out_str = out_str.replace('ァ', ' a')
    out_str = out_str.replace('ィ', ' i')
    out_str = out_str.replace('ゥ', ' u')
    out_str = out_str.replace('ェ', ' e')
    out_str = out_str.replace('ォ', ' o')
    out_str = out_str.replace('ヮ', ' w a')

    out_str = out_str.replace('ヴ', ' b u')

    # 長音の処理
    # for (pattern, replace_str) in JULIUS_LONG_VOWEL:
    #     out_str = pattern.sub(replace_str, out_str)

    # out_str = out_str.replace('o u', 'o:')  # オウ -> オーの音便
    # out_str = out_str.replace('e i', 'e:')  # エイ -> エー
    out_str = out_str.replace('ー', ':')
    out_str = out_str.replace('〜', ':')
    out_str = out_str.replace('−', ':')
    out_str = out_str.replace('-', ':')



    # その他特別な処理
    out_str = out_str.replace('ヲ', ' o')
    out_str = out_str.strip()
    out_str = out_str.replace(':+', ':')

    if is_ret_str:
        return out_str, out_str_orig
    else:
        return out_str
