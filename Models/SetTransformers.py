import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SelfAttention(nn.Module):
    def __init__(self,k, heads):
        super(SelfAttention, self).__init__()
        self.k = k
        self.heads = heads
        self.Wq = nn.Linear(k, heads*k, bias=False)
        self.Wk = nn.Linear(k, heads*k, bias=False)
        self.Wv = nn.Linear(k, heads*k, bias=False)
        self.Wo = nn.Linear(heads*k, k)
    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.Wq(q).view(batch_size, -1, self.heads, self.k).transpose(1,2)
        k = self.Wk(k).view(batch_size, -1, self.heads, self.k).transpose(1,2)
        v = self.Wv(v).view(batch_size, -1, self.heads, self.k).transpose(1,2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / (self.k**0.5)
        qk = F.softmax(qk, dim=-1)
        v = torch.matmul(qk, v)
        v = v.transpose(1,2).contiguous().view(batch_size, -1, self.heads*self.k)
        output = self.Wo(v)
        return output

class Masking(nn.Module):
    def __init__(self,k):
        super(Masking, self).__init__()
        self.k = k
        self.mask = nn.Parameter(torch.tril(torch.ones(k, k)).view(1, 1, k, k))
    def forward(self, x):
        x = x * self.mask
        return x

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super(TransformerBlock, self).__init__()
        self.k = k
        self.heads = heads
        self.attention = SelfAttention(k, heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )
    def forward(self, value, key, query):
        x = self.attention(query, key, value)
        x = x + value
        x = self.norm1(x)
        x = F.relu(x)
        x = self.ff(x)
        x = x + value
        x = self.norm2(x)
        return x

class Masked(nn.Module):
    def __init__(self, k):
        super(Masked, self).__init__()
        self.k = k
        self.mask = nn.Parameter(torch.tril(torch.ones(k, k)).view(1, 1, k, k))
    def forward(self, x):
        x = x * self.mask
        return x
