import matplotlib.pyplot as plt
import numpy as np
import operator
import japanize_matplotlib

def draw_word_char_histgram(
    _dict:dict=None, # a.ka. psylex71.train_data,
    key:str='orth',  # or 'phon'
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
