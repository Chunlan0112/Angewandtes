#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(description='Training LSTM-Wortart-Tagger...')
parser.add_argument("trainfile", help="provide a training file")
parser.add_argument("devfile", help="provide a development file")
parser.add_argument('paramfile',help='provide a parameter file')
parser.add_argument('-n','--num_epochs',type=int,help='times for iteration',default=20)
parser.add_argument('--num_words',type=int,help='vocabulary size',default=10000)
parser.add_argument('--emb_size',type=int,help='embedding size',default=200)
parser.add_argument('--rnn_size',type=int,help='rnn size',default=200)
parser.add_argument('--dropout_rate',type=float,help='dropout_rate',default=0.5)
parser.add_argument('--learning_rate',type=float,help='learning_rate',default=0.5)
args = parser.parse_args()

print(args.num_epochs)

