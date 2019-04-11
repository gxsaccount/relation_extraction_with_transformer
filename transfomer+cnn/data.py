import numpy as np
import torch
import json
import torch.utils.data
__author__ = "Xiang Gao"
# def loadsrcemb():
#     ints={}
#     with open("sgns.baidubaike.bigram-char",'rb')as f:
#         f.readline()
#         for line in f.readlines():
#             tmp=line.decode(encoding = "utf-8").split()
#             key = tmp[0]
#             values=[float(i) for i in tmp[1:]]
#             ints[key]=values
#     return ints 


def json2str():
    max_len = 0
    words=set()
    sentences=[]
    predicates=[]
    en1s=[]
    en2s=[]
    with open("train_data.json",encoding='UTF-8') as f:
        texts = f.readlines()
        for text in texts:
            new_dict = json.loads(text)
            sentence = []
            for term in new_dict['postag']:
                word = term['word']
                words.add(word)
                sentence.append(word)
            for triad in new_dict['spo_list']:
                en1 = triad['object']
                en2 = triad['subject']
                if en1 not in sentence or en2 not in sentence:
                    continue
                en1s.append(sentence.index(triad['object']))
                en2s.append(sentence.index(triad['subject']))
                predicates.append(triad['predicate'])
                max_len = max(max_len,len(sentence))
                sentences.append(sentence)
    print(max_len)
    return words,sentences,en1s,en2s,predicates

def build_sentences():
    words,sentences,en1s,en2s,predicates = json2str()
    word2idx={value:key for key,value in enumerate(words)}
    new_sentences=[[word2idx[word] for word in sentence] for sentence in sentences]
    relation2idx = {value:key for key,value in enumerate(set(predicates))}
    predicates = [[relation2idx[pre]] for pre in predicates]
    # for i in range(sentences):
    return word2idx,new_sentences,en1s,en2s,predicates,relation2idx
    
    # ints = loadsrcemb()
    # new_ints = {}
    # for word in words:
    #     if word in ints:
    #         ints[word]


#build_sentences()



class Dataset(torch.utils.data.Dataset):
    def __init__(self,word2idx,predicates,en1_pos,en2_pos,insts=None):
        self.word2idx = word2idx
        self.insts = insts
        self.en1_pos=en1_pos
        self.en2_pos=en2_pos
        self.idx2word = {idx:word for word,idx in word2idx.items()}
        self.predicates =predicates
    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self.insts)

    @property
    def vocab_size(self):
        ''' Property for vocab size '''
        return len(self.word2idx)

    @property
    def get_word2idx(self):
        ''' Property for word dictionary '''
        return self.word2idx


    @property
    def get_idx2word(self):
        ''' Property for index dictionary '''
        return self.idx2word


    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self.insts[idx],self.en1_pos[idx],self.en2_pos[idx],self.predicates[idx]
