# word2vec のため gensim を使う
import requests
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import os
HOME = os.environ['HOME']


def get_w2v(isColab:bool=False, is2017:bool=True):
	w2v_2017 = {
    	'cbow200': 'http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_cbow.bin.gz',
	    'sgns200': 'http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_sgns.bin.gz',
	    'cbow300': 'http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid300_win20_neg20_sgns.bin.gz',
	    'sgns300': 'http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_cbow.bin.gz'
	}

	w2v_2021 = {
	    'cbow128': { 'id': '1B9HGhLZOja4Xku5c_d-kMhCXn1LBZgDb',
    	            'outfile': '2021_05jawiki_hid128_win10_neg10_cbow.bin.gz'},
	    'sgns128': { 'id': '1OWmFOVRC6amCxsomcRwdA6ILAA5s4y4M',
	                'outfile': '2021_05jawiki_hid128_win10_neg10_sgns.bin.gz'},
	    'cbow200': { 'id': '1JTkU5SUBU2GkURCYeHkAWYs_Zlbqob0s',
    	            'outfile': '2021_05jawiki_hid200_win20_neg20_sgns.bin.gz'}
	}

	#is2017=True

	if isColab:
		#from google_drive_downloader import GoogleDriveDownloader as gdd

		if is2017:
			response = requests.get(w2v_2017['cbow200'])
			fname = w2v_2017['cbow200'].split('/')[-1]
			with open(fname, 'wb') as f:
				f.write(response.content)
		else:
			#訓練済 word2vec ファイルの取得
			(f_id, outfile) = w2v_2021['sgns128']['id'], w2v_2021['sgns128']['outfile']
			gdd.download_file_from_google_drive(file_id=f_id,
    	                                        dest_path=outfile,
        	                                    unzip=False,
            	                                showsize=True)

	if is2017:
		w2v_base = os.path.join(HOME, 'study/2016wikipedia/') if not isColab else '.'
		w2v_file = '2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_cbow.bin.gz'
		w2v_file = os.path.join(w2v_base, w2v_file)
	else:
		w2v_base = os.path.join(HOME, 'study/2019attardi_wikiextractor.git/wiki_texts/AA') if isMac else '.'
		w2v_file = '2021_05jawiki_hid128_win10_neg10_sgns.bin'

	w2v = KeyedVectors.load_word2vec_format(w2v_file,encoding='utf-8',unicode_errors='replace',	binary=True)

	return w2v
