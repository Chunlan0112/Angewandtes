#!/usr/bin/env python3
import torch
import torch.nn as nn


class TaggerModel(nn.Module):
    def __init__(self, numWords, numTags, embSize, rnnSize, dropoutRate):
        super().__init__()
        self.embedding = nn.Embedding(numWords, embSize)
        self.dropout = nn.Dropout(p=dropoutRate)
        self.lstm = nn.LSTM(embSize, rnnSize, bidirectional=True)
        self.linear = nn.Linear(rnnSize*2, numTags)

    def forward(self,sentence):
        embeds = self.embedding(sentence)
        embeds = self.dropout(embeds)
        embeds = torch.unsqueeze(embeds,dim=1)
        out,_ = self.lstm(embeds)
        out = torch.squeeze(out, dim=1)
        out = self.linear(self.dropout(out))
        return out