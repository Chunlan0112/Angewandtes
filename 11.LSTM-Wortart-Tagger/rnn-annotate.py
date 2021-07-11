#!/usr/bin/env python3
import torch
from Data import Data
import argparse


parser = argparse.ArgumentParser(description='Using model to annotate new sentence...')
parser.add_argument("paramfile", help="provide a parameter file")
parser.add_argument('sentence',help="the file to be annotated")
args = parser.parse_args()

data = Data(args.paramfile+".io" )  # read the symbol mapping tables
model = torch.load(args.paramfile+".rnn")  # read the model

sentences = data.sentences(args.sentence)
for sentence in sentences:
    print(' '.join(sentence))
    with torch.no_grad():
        sentence = data.words2IDs(sentence)
        tag_score = model(sentence)
        pred_tags = torch.argmax(tag_score, dim=1)
        tags = data.IDs2tags(pred_tags)
        print(' '.join(tags),'\n')
