#!/usr/bin/env python3
import torch
from Data import Data
import argparse

# Shuzhou Yuan Matrikelnummer: 11993161

parser = argparse.ArgumentParser(description='Using model to annotate new testfile...')
parser.add_argument("paramfile", help="provide a parameter file")
parser.add_argument('testfile',help="the file to be annotated")
args = parser.parse_args()

data = Data(args.paramfile+".io" )  # read the symbol mapping tables
model = torch.load(args.paramfile+".rnn")  # read the model

sentences = data.sentences(args.testfile)
for sentence in sentences:
    print('Sentence: '+' '.join(sentence))
    with torch.no_grad():
        f,b = data.words2IDvecs(sentence)
        tag_score = model(f,b)
        pred_tags = torch.argmax(tag_score, dim=1)
        tags = data.IDs2tags(pred_tags)
        print('Tags: '+' '.join(tags)+'\n')
