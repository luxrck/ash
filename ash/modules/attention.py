import math

import torch
from torch import nn
import torch.nn.functional as F

from entmax import sparsemax, entmax15, entmax_bisect


class ScaledDotAttention(nn.Module):
    def __init__(self, d_hidden, with_sparse=False, dropout_p=0.3):
        super().__init__()
        self.with_sparse = with_sparse
        self.d_hidden = d_hidden
        self.proj_q = nn.Linear(d_hidden, d_hidden)
        self.proj_k = nn.Linear(d_hidden, d_hidden)
        self.proj_v = nn.Linear(d_hidden, d_hidden)
        
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Sequential(
                    nn.Linear(d_hidden, d_hidden),
                    nn.Dropout(p=dropout_p))
    
    def forward(self, Q, K, V, scale=-1, with_projection=True, with_output=False, q_mask=None, k_mask=None):
        # import pdb; pdb.set_trace()
        if with_projection:
            Q = self.proj_q(Q)
            K = self.proj_k(K)
            V = self.proj_v(V)

        if scale <= 0:
            scale = math.sqrt(self.d_hidden)
        
        scale = 1
        
        # [bs, q_size, dim] * [bs, k_size, dim] -> [bs, q_size, k_size]
        # import pdb; pdb.set_trace()
        e = Q.matmul(K.transpose(-1, -2)) / scale

        masked = -2**14 if Q.dtype == torch.float16 else -2**31
        
        if k_mask is not None:
            # k_mask: [bs, k_len] -> [bs, 1, k_len]
            k_mask = k_mask.unsqueeze(-2)
            e.masked_fill_(k_mask == 0, masked)
        
        # a = F.softmax(e, dim=-1)
        if self.with_sparse:
            a = sparsemax(e, dim=-1)
        else:
            a = F.softmax(e, dim=-1)
        a = self.dropout(a)
        a = a.matmul(V)

        if with_output:
            a = self.out(a)

        if q_mask is not None:
            # q_mask: [bs, q_len] -> [bs, .. , q_len]
            q_mask = q_mask.expand(a.shape[:-1])
            a[q_mask == 0] = 0.
        
        return a, e



class MultiHeadAttention(nn.Module):
    # d_hidden: 输入和输出的向量维度
    def __init__(self, d_hidden, n_head=8, dropout_p=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_hidden = d_hidden // n_head
        self.n_head = n_head

        assert self.d_hidden * self.n_head == d_hidden

        self.linear_q = nn.Linear(d_hidden, d_hidden)    # d_q == d_k
        self.linear_k = nn.Linear(d_hidden, d_hidden)
        self.linear_v = nn.Linear(d_hidden, d_hidden)
        
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(d_hidden, d_hidden)
    
    # (b, s, d)
    def forward(self, q, k, v, scale=-1, with_projection=True, with_output=False, q_mask=None, k_mask=None):
        if scale <= 0:
            scale = math.sqrt(self.d_hidden)
        
        bs = q.size(0)
        n_head = self.n_head
        dim = self.d_hidden

        if with_projection:
            q = self.linear_q(q)
            k = self.linear_q(k)
            v = self.linear_q(v)

        q = q.view(bs, -1, n_head, dim).transpose(1, 2)#.contiguous().permute(2, 0, 1, 3).contiguous().view(-1, s_q, dim)
        k = k.view(bs, -1, n_head, dim).transpose(1, 2)#.contiguous().permute(2, 0, 1, 3).contiguous().view(-1, s_k, dim)
        v = v.view(bs, -1, n_head, dim).transpose(1, 2)#.contiguous().permute(2, 0, 1, 3).contiguous().view(-1, s_v, dim)

        # [bs, nh, q_size, dim] * [bs, nh, k_size, dim] -> [bs, nh, q_size, k_size]
        # import pdb; pdb.set_trace()
        e = q.matmul(k.transpose(-1, -2)) / scale

        masked = -2**14 if q.dtype == torch.float16 else -2**31
        
        if k_mask is not None:
            # k_mask: [bs, k_len] -> [bs, 1, k_len]
            k_mask = k_mask.unsqueeze(-2)
            e.masked_fill_(k_mask == 0, masked)
        
        a = F.softmax(e, dim=-1)
        a = self.dropout(a)
        a = a.matmul(v)
        # a = a.view(n_head, bs, s_q, dim).contiguous().permute(1, 2, 0, 3).contiguous().view(bs, s_q, -1)
        a = a.transpose(1, 2).contiguous().view(bs, -1, n_head * dim)

        if with_output:
            a = self.out(a)

        if q_mask is not None:
            # q_mask: [bs, q_len] -> [bs, .. , q_len]
            q_mask = q_mask.expand(a.shape[:-1])
            a[q_mask == 0] = 0.
        
        return a, e