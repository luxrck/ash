import torch
from torch import nn
import torch.nn.functional as F

from entmax import sparsemax, entmax15, entmax_bisect

from .attention import ScaledDotAttention, MultiHeadAttention



#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021-01-13 10:55
# @Author  : NingAnMe <ninganme@qq.com>

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class LabelSmoothing(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self):
        super(LabelSmoothing, self).__init__()
    
    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            # FIXME: 为什么要 n_classes - 2
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, output, target, smoothing=0.1, ignore_index=-100, ignore_mask=None, reduction="mean"):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # import pdb; pdb.set_trace()
        target_smoothing = LabelSmoothing._smooth_one_hot(target, output.size(-1), smoothing)
        target_smoothing.masked_fill_((target == ignore_index).unsqueeze(1), 0)
        if ignore_mask is not None:
            target_smoothing.masked_fill_(ignore_mask == 0, 0)

        #torch.abs(F.kl_div(w_col.squeeze(-1).log_softmax(dim=-1), gw_col, reduction="batchmean"))
        # import pdb; pdb.set_trace()
        # output = sparsemax(output, dim=-1).log()
        # output[output == float('-inf')] = - 2 ** 31
        return torch.abs(F.kl_div(output.log_softmax(dim=-1), target_smoothing, reduction=reduction))



class PointerPredicator(nn.Module):
    def __init__(self, embedding, d_hidden, dropout_p=0.3, smoothing=0.1, pointers=[]):
        super().__init__()
        self.d_hidden = d_hidden
        self.dropout_p = dropout_p
        self.cls_embedding = embedding

        self.cross_entropy = LabelSmoothing()

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.layer_norm1 = nn.LayerNorm(self.d_hidden)
        # self.cls_embedding = nn.Embedding(len(vocab), self.d_hidden, padding_idx=padding_idx)

        self.pointer_base = self.vocab_size = self.cls_embedding.num_embeddings

        # self.ptr_attns = nn.ModuleDict([
        #     (task, Attention(self.d_hidden)) for task in pointers
        #     ])

        self.ptr_attns = nn.ModuleList([ScaledDotAttention(self.d_hidden, dropout_p=dropout_p) for _ in range(len(pointers))])
        # self.ptr_attns = nn.ModuleList([nn.MultiheadAttention(self.d_hidden, num_heads=8, dropout=self.dropout_p) for _ in range(len(pointers))])
        # self.ptr_attn0 = Attention(self.d_hidden, dropout_p=dropout_p)
        # self.ptr_attn1 = Attention(self.d_hidden, dropout_p=dropout_p)

        self.cls_predicator = nn.Sequential(
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.Dropout(p=dropout_p),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(self.d_hidden, self.cls_embedding.num_embeddings)
        )
        self.ptr_predicator = nn.Sequential(
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.Dropout(p=dropout_p),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(self.d_hidden, 1)
        )

    # pointers: [bs, n_ptr, dim]
    # x: [bs, q_len]
    def embedding(self, x, pointers=None, debug=None, ex=None):
        if pointers is None:
            return self.cls_embedding(x)

        bs = x.size(0)
        n_x = x.size(1)
        n_ptrs = [ptr.size(1) for ptr in pointers]
        dim = pointers[0].size(2)
        x_embs = torch.zeros(bs, n_x, dim).cuda()

        for b in range(bs):
            for i, x_idx in enumerate(x[b]):
                if x_idx >= self.pointer_base:
                    p_idx = -1
                    x_idx = x_idx.item()        # <------------ 注意这里！！！！！！先detach，不然会修改原有的tensor
                    x_idx -= self.pointer_base
                    for j, n_ptr in enumerate(n_ptrs):
                        if x_idx < n_ptr:
                            p_idx = j; break
                        x_idx -= n_ptr
                    assert p_idx != -1
                    try:
                        if debug:
                            assert x_idx == debug[b][i]
                    except:
                        import pdb; pdb.set_trace()
                    e = pointers[p_idx][b, x_idx, :]
                else:
                    e = self.cls_embedding(x_idx)
                # import pdb; pdb.set_trace()
                x_embs[b, i, :] = e.view(-1)
        return x_embs

    # pointers: [bs, p_cnt, dim]
    # x: [bs, x_cnt, dim]
    def forward(self, x, mask, pointers, ex=None):

        c = [x]
        ptr_scores = []

        # import pdb; pdb.set_trace()
        for i, attention in enumerate(self.ptr_attns):
            ptr, ptr_mask = pointers[i]
            # a, e = self.ptr_attns[i](x.transpose(0, 1), ptr.transpose(0, 1), ptr.transpose(0, 1), key_padding_mask=(ptr_mask ^ 1).bool())
            # a = a.transpose(0, 1)
            a, e = self.ptr_attns[i](x, ptr, ptr, with_projection=True, with_output=False, q_mask=mask, k_mask=ptr_mask)
            # a = self.dropout1(a)
            c.append(a)
            ptr_scores.append(e)

        c = torch.stack(c, dim=0).sum(dim=0)
        c = self.layer_norm1(c)
        c = self.dropout2(c)

        cls_score = self.cls_predicator(c)

        # import pdb; pdb.set_trace()
        y_prob = torch.cat([cls_score] + ptr_scores, dim=-1)

        # import pdb; pdb.set_trace()
        cls_mask = torch.ones(x.size(0), self.cls_embedding.num_embeddings, device=x.device, dtype=torch.long)
        y_prob_mask = torch.cat([cls_mask] + [mask for _, mask in pointers], dim=-1)
        return y_prob, y_prob_mask

    def loss(self, predict, label, padding_idx=0, padding_mask=None, debug=None):
        bs = predict.size(0)
        ncls = predict.size(2)
        loss = 0.
        for b in range(bs):
            # import pdb; pdb.set_trace()
            n_ptr = (label[b] != 0).sum().item()
            loss += self.cross_entropy(
                predict[b, :n_ptr-1, :].contiguous().view(-1, ncls),
                label[b, 1:n_ptr].contiguous().view(-1),
                ignore_index=padding_idx,
                ignore_mask =padding_mask[b],
                reduction='sum',
                )
        loss /= bs
        # loss = F.cross_entropy(predict[:, :-1, :].contiguous().view(-1, ncls), label[:, 1:].contiguous().view(-1), ignore_index=padding_idx)
        return [loss]

# F.cross_entropy(predict[b, :-1, :].contiguous().view(-1, ncls), label[b, 1:].contiguous().view(-1), ignore_index=0, reduction='sum')