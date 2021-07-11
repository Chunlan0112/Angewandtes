#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from Data import Data
from TaggerModel import TaggerModel
import random
import argparse

# Shuzhou Yuan Matrikelnummer: 11993161

parser = argparse.ArgumentParser(description='Training LSTM-Wortart-Tagger...')
parser.add_argument("trainfile", help="provide a training file")
parser.add_argument("devfile", help="provide a development file")
parser.add_argument('paramfile',help='provide a parameter file')
parser.add_argument('-e','--num_epochs',type=int,help='times for iteration',default=10)
parser.add_argument('-w','--wordlength',type=int,help='word length',default=10)
parser.add_argument('-m','--emb_size',type=int,help='embedding size',default=300)
parser.add_argument('-r1','--rnn_size1',type=int,help='rnn size for word embedding',default=200)
parser.add_argument('-r2','--rnn_size2',type=int,help='rnn size for BiLSTM',default=200)
parser.add_argument('-d','--dropout_rate',type=float,help='dropout_rate',default=0.3)
parser.add_argument('-l','--learning_rate',type=float,help='learning_rate',default=0.001)
args = parser.parse_args()

data = Data(args.trainfile,args.devfile,args.wordlength)
tagger = TaggerModel(data.numTags,data.numSym,args.emb_size,args.rnn_size1,args.rnn_size2,args.dropout_rate)
data.store_parameters(args.paramfile+".io")
print('data file is saved!')

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(tagger.parameters(), lr=args.learning_rate)

precision = 0
for x in range(args.num_epochs):
    tagger = tagger.cuda()
    print(x+1,' times training...')
    tagger.train(True)
    train_sentences = data.trainSentences
    random.shuffle(train_sentences)
    n = 0
    for sent, tags in train_sentences:
        n+=1
        f,b = data.words2IDvecs(sent)
        f,b = f.cuda(),b.cuda()
        tag_tensor = data.tags2IDs(tags).cuda()

        tagger.zero_grad()

        score = tagger(f,b)
        loss = loss_function(score, tag_tensor)

        loss.backward()
        optimizer.step()

    print(n,' sentences are trained...')

    tagger.train(False)
    with torch.no_grad():
        correct_num = 0
        all_num = 0
        for sent, tags in data.devSentences:
            f,b = data.words2IDvecs(sent)
            f,b = f.cuda(), b.cuda()
            tag_tensor = data.tags2IDs(tags).cuda()
            tag_score = tagger(f,b)
            pred_tags = torch.argmax(tag_score, dim=1)
            for pred_tag,true_tag in zip(data.IDs2tags(pred_tags),tags):
                all_num+=1
                if pred_tag == true_tag:
                    correct_num +=1
        precision_dev = correct_num/all_num
        print(x+1,' iteration : precision ',precision_dev)
        if precision_dev > precision:
            precision = precision_dev
            torch.save(tagger.cpu(), args.paramfile + '.rnn')
            print('model is saved!')