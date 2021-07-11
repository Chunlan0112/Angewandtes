#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from Data import Data
from TaggerModel import TaggerModel
import random
import argparse


parser = argparse.ArgumentParser(description='Training LSTM-Wortart-Tagger...')
parser.add_argument("trainfile", help="provide a training file")
parser.add_argument("devfile", help="provide a development file")
parser.add_argument('paramfile',help='provide a parameter file')
parser.add_argument('-e','--num_epochs',type=int,help='times for iteration',default=20)
parser.add_argument('-v','--num_words',type=int,help='vocabulary size',default=10000)
parser.add_argument('-m','--emb_size',type=int,help='embedding size',default=200)
parser.add_argument('-r','--rnn_size',type=int,help='rnn size',default=200)
parser.add_argument('-d','--dropout_rate',type=float,help='dropout_rate',default=0.5)
parser.add_argument('-l','--learning_rate',type=float,help='learning_rate',default=0.001)
args = parser.parse_args()

data = Data(args.trainfile,args.devfile,args.num_words)
tagger = TaggerModel(data.numWords, data.numTags, args.emb_size, args.rnn_size, args.dropout_rate)
data.store_parameters(args.paramfile+".io")
print('data file is saved!')

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(tagger.parameters(), lr=args.learning_rate)

precision = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tagger = tagger.to(device)
for x in range(args.num_epochs):
    print(x+1,' iteration training...')
    tagger.train(True)
    train_sentences = data.trainSentences
    random.shuffle(train_sentences)
    for sent, tags in train_sentences:
        sent_tensor = data.words2IDs(sent).cuda()
        tag_tensor = data.tags2IDs(tags).cuda()

        tagger.zero_grad()
        tag_score = tagger(sent_tensor)
        loss = loss_function(tag_score, tag_tensor)

        loss.backward()
        optimizer.step()

    tagger.train(False)
    with torch.no_grad():
        correct_num = 0
        all_num = 0
        for sent, tags in data.devSentences:
            sents_tensor = data.words2IDs(sent).cuda()
            tags_tensor = data.tags2IDs(tags).cuda()
            tag_score = tagger(sents_tensor)
            pred_tags = torch.argmax(tag_score, dim=1)
            for pred_tag,true_tag in zip(data.IDs2tags(pred_tags),tags):
                all_num+=1
                if pred_tag == true_tag:
                    correct_num += 1
        precision_dev = correct_num/all_num
        print(x+1,' iteration : presicion ',precision_dev)
        if precision_dev > precision:
            precision = precision_dev
            torch.save(tagger.cpu(), args.paramfile + '.rnn')
            print('model is saved!')