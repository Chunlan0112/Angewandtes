#!/usr/bin/env python3
from collections import Counter
import torch
import pickle

class Data:
    def __init__(self, *args):
        if len(args) == 1:
            self.init_test(*args)
        else:
            self.init_train(*args)

    def init_test(self,filename):
        with open (filename,'rb') as file:
            map_table = pickle.load(file)
        self.word_id = map_table['word_id']
        self.id_tag = map_table['id_tag']

    def init_train(self,trainfile, devfile,numWords):
        self.numWords = numWords + 1
        self.trainSentences = self.readData(trainfile)
        self.devSentences = self.readData(devfile)
        self.tag_id = {tag:id for id,tag in enumerate({tag for _,tags in self.trainSentences for tag in tags},start=1)}
        self.id_tag = {id:tag for tag,id in self.tag_id.items()}
        self.id_tag[0] = 'UNKNOWN'
        self.word_id = self.reorder_from_freqDict()
        self.numTags = len(self.id_tag)

    def readData(self, file):
        sentences, words, tags = [], [], []
        with open(file,encoding='utf-8') as f:
            for line in f:
                if line != '\n':  # add word and tag to sent_1 and tags_1
                    word, tag = line.split()
                    tags.append(tag)
                    words.append(word)
                else:  # add sentence and tags to sent and tags
                    sentences.append((words,tags))
                    words, tags = [], []  # clean list for next sentence
            if words and tags:  # add the last sentence
                sentences.append((words,tags))
        return sentences

    def reorder_from_freqDict(self):
        wordFreq = Counter([word for words, tags in self.trainSentences for word in words])
        words, _ = zip(*wordFreq.most_common(self.numWords-1))
        word_id = {word: i for i, word in enumerate(words, 1)}
        return word_id

    def words2IDs(self, words):
        word2id = [self.word_id.get(word, 0) for word in words]
        return torch.tensor(word2id, dtype=torch.int)

    def tags2IDs(self,tags):
        tag2id = [self.tag_id.get(tag,0) for tag in tags]
        return torch.tensor(tag2id, dtype=torch.long)

    def IDs2tags(self,ids):
        return [self.id_tag[id.item()] for id in ids]

    def store_parameters(self,filename):
        map_table = dict()
        map_table['word_id'] = self.word_id
        map_table['id_tag'] = self.id_tag
        with open (filename,'wb') as file:
            pickle.dump(map_table,file)

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
