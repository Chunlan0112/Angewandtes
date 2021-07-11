#!/usr/bin/env python3
import torch
import torch.nn as nn

# Shuzhou Yuan Matrikelnummer: 11993161


class TaggerModel(nn.Module):
    def __init__(self, numTags, numSym, embSize, rnnSize1, rnnSize2, dropoutRate):
        super().__init__()
        self.emb = nn.Embedding(numSym,embSize)
        self.emb_lstm = nn.LSTM(embSize,rnnSize1,batch_first=True)
        self.dropout = nn.Dropout(p=dropoutRate)
        self.lstm = nn.LSTM(rnnSize1*2,rnnSize2,bidirectional=True)
        self.linear = nn.Linear(rnnSize2*2,numTags)

    def forward(self,suffix,preffix):
        suffix = self.emb(suffix)
        preffix = self.emb(preffix)
        forward,_ = self.emb_lstm(self.dropout(suffix))
        backward,_ = self.emb_lstm(self.dropout(preffix))
        embeds = torch.cat([forward[:,-1,:],backward[:,-1,:]],dim=1)
        embeds = self.dropout(embeds)
        embeds = torch.unsqueeze(embeds,dim=1)
        out,_ = self.lstm(embeds)
        out = torch.squeeze(out, dim=1)
        tag_score = self.linear(self.dropout(out))
        return tag_score

