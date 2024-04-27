import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class decoder1(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder1, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU(),
            torch.nn.Linear(nhid1, nfeat),
            torch.nn.ReLU()
        )
    def forward(self, emb):
        x = self.decoder(emb)
        return x

class AttentionWide(nn.Module):
    def __init__(self, emb, p = 0.2, heads=8, mask=False):
        super().__init__()

        self.emb = emb
        self.heads = heads
        self.dropout = nn.Dropout(p)
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, y):
        b = 1
        t, e = x.size()
        h = self.heads

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.dropout(self.toqueries(y)).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)
        dot = F.softmax(dot, dim=2)
        self.attention_weights = dot
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class stMCFN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(stMCFN, self).__init__()
        self.GCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.GCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.dec = decoder1(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.act = F.relu
        self.attn2 = AttentionWide(nhid2, heads=8)      # 8

    def forward(self, x, sadj, fadj):
        emb1 = self.GCN1(x, sadj)
        emb2 = self.GCN2(x, fadj)
        emb = (self.attn2(emb1, emb2)).squeeze(0)+emb1
        H = self.dec(emb)
        H1 = self.dec(emb1)
        H2 = self.dec(emb2)

        return emb1, emb2, emb, H1, H2, H
