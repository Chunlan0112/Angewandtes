#!/usr/bin/env python3
from collections import defaultdict
import torch
import pickle
from collections import Counter

# Shuzhou Yuan Matrikelnummer: 11993161


class Data:
    def __init__(self, *args):
        if len(args) == 1:
            self.init_test(*args)
        else:
            self.init_train(*args)

    def init_test(self,filename):
        with open(filename, 'rb') as file:
            map_table = pickle.load(file)
        self.sym_id = map_table['sym_id']
        self.id_tag = map_table['id_tag']
        self.wordLength = map_table['wordLength']

    def init_train(self,trainfile, devfile, wordLength):
        self.trainSentences = self.readData(trainfile)
        self.devSentences = self.readData(devfile)
        self.tag_id = {tag: id for id, tag in
                       enumerate({tag for _, tags in self.trainSentences for tag in tags}, start=1)}
        self.id_tag = {id: tag for tag, id in self.tag_id.items()}
        self.id_tag[0] = 'UNKNOWN'
        self.sym_id = self.Indextabelle()
        self.numTags = len(self.id_tag)
        self.wordLength = wordLength
        self.numSym = len(self.sym_id)+1

    def readData(self, file):
        sentences, words, tags = [], [], []
        with open(file, encoding='utf-8') as f:
            for line in f:
                if line != '\n':  # add word and tag to sent_1 and tags_1
                    word = line.split()[0]
                    tag = line.split()[1]
                    tags.append(tag)
                    words.append(word)
                else:  # add testfile and tags to sent and tags
                    sentences.append((words, tags))
                    words, tags = [], []  # clean list for next testfile
            if words and tags:  # add the last testfile
                sentences.append((words, tags))
        return sentences

    def Indextabelle(self):
        symFreq = Counter([sym for sent,_ in self.trainSentences for word in sent for sym in word])
        sym_id = {sym:id for id,sym in enumerate([s for s,f in symFreq.items() if f > 1],start=1)}
        return sym_id

    def words2IDvecs(self,words):
        suffix, prefix = [],[]
        for word in words:
            if len(word) < self.wordLength:
                suff = word.rjust(self.wordLength)
                pref = word.ljust(self.wordLength)
                #suff = ' '*(self.wordLength-len(word))+word
                #pref = word + ' '*(self.wordLength-len(word))
            else:
                suff = word[-self.wordLength:]
                pref = word[:self.wordLength]
            suffix_id = [self.sym_id.get(l,0) for l in suff]
            prefix_id = [self.sym_id.get(l,0) for l in pref[::-1]]
            prefix.append(prefix_id)
            suffix.append(suffix_id)
        return torch.tensor(suffix,dtype=torch.long),torch.tensor(prefix,dtype=torch.long)

    def tags2IDs(self,tags):
        tag2id = [self.tag_id.get(tag,0) for tag in tags]
        return torch.tensor(tag2id, dtype=torch.long)

    def IDs2tags(self,ids):
        return [self.id_tag[id.item()] for id in ids]

    def store_parameters(self,filename):
        map_table = dict()
        map_table['sym_id'] = self.sym_id
        map_table['id_tag'] = self.id_tag
        map_table['wordLength'] = self.wordLength
        with open(filename, 'wb') as file:
            pickle.dump(map_table, file)

    def sentences(self,filename):
        wordlist = []
        with open(filename,'r') as file:
            for tok in file:
                if tok !='\n':
                    wordlist.append(tok.strip())
                else:
                    yield wordlist
                    wordlist.clear()
            yield wordlist