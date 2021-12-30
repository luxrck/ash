import os
import re
import sys
import functools
import inspect
import random
import argparse
from collections import defaultdict, deque
from typing import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.cuda import amp

from torch import nn, optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import transformers

from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from modeling_vilbert import VilBertModel, VilBertConfig
from tokenization_vilbert import VilBertTokenizer

from data import mask, pad

import h5py


@dataclass
class DataCollatorForVBertMatch:
    tokenizer: PreTrainedTokenizerBase
    device = "cpu"
    pad_to_multiple_of = None

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        fields = set()
        # Handle dict or lists with proper padding and conversion to tensor.
        # if isinstance(examples[0], (dict, BatchEncoding)):
        fields.update(examples[0].keys())
        # import pdb; pdb.set_trace()
        try:
            batch = self.tokenizer.pad(examples, pad_to_multiple_of=self.pad_to_multiple_of)
        except Exception as err:
            print([e['photo_id'] for e in examples])
            raise err

        device = self.device
        batch["image_attention_mask"] = mask([len(e["image_embeds"]) for e in examples], device=device)

        for f in fields:
            if 'label' in f or f in {"input_ids", "attention_mask", "special_tokens_mask", "image_embeds", "image_attention_mask", "tag_input_ids", "tag_attention_mask", "tag_special_tokens_mask"}:
                batch[f] = pad(batch[f], device=device)

        if batch['input_ids'].size(1) >= 512:
            import pdb; pdb.set_trace()
        return batch



class VBertMatchBaseDataset(Dataset):
    def __init__(self, filename, tokenizer, offset=0, size=None, with_neg=False, only_neg=False, **kw):
        self.df = h5py.File(filename, 'r', swmr=True)
        self.cache_ = []
        self.offset_ = offset
        self.size_ = size
        self.tokenizer_ = tokenizer
        self.with_neg_ = with_neg
        self.only_neg_ = only_neg
        self.has_vlm_label = 'vlm_label' in self.df
    def __len__(self):
        # 75847681
        # return 250
        if self.size_ is not None:
            return self.size_
        return 71501100
        # return len(self.df['photo_id']) - 200000  # Pos Examples + Neg Examples (1:1)
    def __getitem__(self, k):
        CLS = 2
        SEP = 3
        TAG = 24074 # [#]

        def get_neg_idx(k):
            # idx = random.randint(max(self.offset_, k-200000), min(k+200000, self.offset_ + len(self)))
            idx = random.randint(0, len(self))
            # idx = random.randint(0, 75000000)
            idx += self.offset_
            if idx == k:
                return get_neg_idx(k)
            return idx

        k += self.offset_

        # photo_id = self.df['photo_id'][k]
        # if photo_id == 0:
        #     next_k = get_neg_idx(k)
        #     return self.__getitem__(next_k)

        # text_length = self.df["text_length"][k]
        # import pdb; pdb.set_trace()
        text_ids = self.df["text_id"][k]
        text_length = (text_ids != 0).sum()

        # if text_length < 2:
            # text_length = 2
            # text_ids[0] = CLS
            # text_ids[1] = SEP
        
        if text_length > 78:
            text_length = 78
        
        text_ids = [CLS] + text_ids[:text_length].tolist() + [SEP]
        text_length += 2
        
        attention_mask = [1] * text_length
        special_tokens_mask = [1] + [0] * (text_length-2) + [1]

        tag_text_ids = self.df["tag_id"][k]
        tag_length = (tag_text_ids != 0).sum()

        # if tag_length < 2:
        #     tag_length = 2
        #     tag_text_ids[0] = CLS
        #     tag_text_ids[1] = SEP

        if tag_length > 78:
            tag_length = 78
        
        tag_text_ids = tag_text_ids[:tag_length]
        tag_text_ids = [CLS] + tag_text_ids.tolist() + [SEP]
        tag_length += 2
        
        tag_attention_mask = [1] * tag_length
        tag_special_tokens_mask = [1] + [0] * (tag_length-2) + [1]

        image_embed = self.df["ft"][k].tolist()
        tag_label = self.df["label"][k].item()
        
        return {
            # "photo_id": photo_id,
            # "negative_photo_id": 0,
            "input_ids": text_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "tag_input_ids": tag_text_ids,
            "tag_attention_mask": tag_attention_mask,
            "tag_special_tokens_mask": tag_special_tokens_mask,
            "image_embeds": [image_embed],   # FIXME: 当前image patch size == 1
            "text_length": text_length,
            "image_length": 1,
            # "text": text,
            # "tag_idx": tag_sep,
            # "vlm_label": vlm_label,
            "tag_label": tag_label,
        }


@dataclass
class DataCollatorForVBertMatchBase:
    tokenizer: PreTrainedTokenizerBase
    device = "cpu"
    pad_to_multiple_of = None

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        fields = set()
        # Handle dict or lists with proper padding and conversion to tensor.
        # if isinstance(examples[0], (dict, BatchEncoding)):
        fields.update(examples[0].keys())
        # import pdb; pdb.set_trace()
        try:
            batch = self.tokenizer.pad(examples, pad_to_multiple_of=self.pad_to_multiple_of)
        except Exception as err:
            print([e['photo_id'] for e in examples])
            raise err

        device = self.device
        batch["image_attention_mask"] = mask([len(e["image_embeds"]) for e in examples], device=device)

        for f in fields:
            if 'label' in f or f in {"input_ids", "attention_mask", "special_tokens_mask", "image_embeds", "image_attention_mask", "tag_input_ids", "tag_attention_mask", "tag_special_tokens_mask"}:
                batch[f] = pad(batch[f], device=device)

        return batch



class MLPEncoder(nn.Module):
    def __init__(self, embeddings):
        super().__init__()

        self.embeddings = embeddings
        self.pooler = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(768, 768),
            # nn.ReLU(),
            # nn.Dropout(0.1)
        )
    
    def forward(self, b):
        tag_input_ids = b.tag_input_ids
        tag_attention_mask = b.tag_attention_mask
        # import pdb; pdb.set_trace()
        tag_text_embeds = self.embeddings(tag_input_ids)

        bs = tag_text_embeds.size(0)
        tag_text_lengths = tag_attention_mask.eq(1).sum(-1).view(-1)
        tag_embeds = []
        for i in range(bs):
            emb = tag_text_embeds[i, :tag_text_lengths[i]].mean(0)
            tag_embeds.append(emb)
        # import pdb; pdb.set_trace()
        tag_embeds = torch.stack(tag_embeds)
        # tag_embeds = self.pooler(tag_embeds)

        return tag_embeds



class VBertMatchBase(nn.Module):
    def __init__(self, config, vbert, tbert, dropout_p=0.1):
        super().__init__()

        self.d_hidden = 768

        self.vbert = vbert
        self.tbert = tbert
        # self.tbert = MLPEncoder(self.tbert)
        
        # self.tmatcher = nn.Sequential(
        #     # nn.Linear(self.d_hidden * 3, 1),    # tag, tit, img_last, img_first
        #     nn.Linear(self.d_hidden * 3, self.d_hidden),    # tag, tit, img_last, img_first
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_p),
        #     # nn.LayerNorm(normalized_shape=self.d_hidden),
        #     # nn.Linear(self.d_hidden, 1)
        # )
        
        # self.imatcher = nn.Sequential(
        #     # nn.Linear(self.d_hidden * 3, 1),    # tag, tit, img_last, img_first
        #     nn.Linear(self.d_hidden * 3, self.d_hidden),    # tag, tit, img_last, img_first
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_p),
        #     # nn.LayerNorm(normalized_shape=self.d_hidden),
        #     # nn.Linear(self.d_hidden, 1)
        # )
        
        # self.fuse = nn.Sequential(
        #     nn.Linear(self.d_hidden * 3, 1),    # tag, tit, img_last, img_first
        #     # nn.Linear(self.d_hidden * 2, 1),    # tag, tit, img_last, img_first
        # )

        self.vpooler = nn.Sequential(
            # nn.LayerNorm(normalized_shape=self.d_hidden),
            # nn.Dropout(p=dropout_p),
            # nn.Linear(self.d_hidden, self.d_hidden),
            # nn.ReLU(),
            nn.Dropout(p=dropout_p),
            # nn.Linear(self.d_hidden, 128),
            # nn.ReLU(),
            # nn.LayerNorm(normalized_shape=self.d_hidden),
            # nn.Dropout(p=dropout_p),
            # nn.Linear(self.d_hidden, self.d_hidden),
        )

        # self.seq_relationship = nn.Linear(self.d_hidden, 2)

        self.ipooler = nn.Sequential(
            # nn.Linear(self.d_hidden, self.d_hidden),
            # nn.ReLU(),
            nn.Dropout(p=dropout_p),
            # nn.LayerNorm(normalized_shape=self.d_hidden),
            # nn.Linear(self.d_hidden, 128),
            # nn.ReLU(),
            # nn.LayerNorm(normalized_shape=self.d_hidden),
            # nn.Dropout(p=dropout_p),
            # nn.Linear(self.d_hidden, self.d_hidden),
        )

        self.tpooler = nn.Sequential(
            # nn.Linear(self.d_hidden, self.d_hidden),
            # nn.ReLU(),
            nn.Dropout(p=dropout_p),
            # nn.LayerNorm(normalized_shape=self.d_hidden),
            # nn.Linear(self.d_hidden, 128),
            # nn.ReLU(),
            # nn.Linear(self.d_hidden, self.d_hidden),
        )

        # self.bpooler = nn.Sequential(
        #     # nn.Linear(self.d_hidden, self.d_hidden),
        #     # nn.ReLU(),
        #     nn.Dropout(p=dropout_p),
        #     # nn.LayerNorm(normalized_shape=self.d_hidden),
        #     # nn.Linear(self.d_hidden, 128),
        #     # nn.ReLU(),
        #     # nn.Linear(self.d_hidden, self.d_hidden),
        # )

    def match(self, b, with_img=True, fuse="1"):
        
        encoded = self.vbert(
            text_ids=b.input_ids,
            text_attention_mask=b.attention_mask,
            image_embeds=(b.image_embeds if with_img else torch.zeros_like(b.image_embeds.zero_())),
            # image_attention_mask=(b.image_attention_mask if with_img else None),
            return_dict=True,
        )
        # import pdb; pdb.set_trace()
        # encoded_tag = self.tbert(
        #     text_ids=b.tag_input_ids,
        #     text_attention_mask=b.tag_attention_mask,
        #     # image_embeds=b.image_embeds,
        #     image_embeds=(b.image_embeds if with_img and fuse == "1" else torch.zeros_like(b.image_embeds)),
        #     # image_embeds=b.image_embeds,
        #     # image_attention_mask=(b.image_attention_mask if with_img else None),
        #     return_dict=True,
        # )

        encoded_tag = self.tbert(
            input_ids=b.tag_input_ids,
            attention_mask=b.tag_attention_mask,
            # image_embeds=b.image_embeds.zero_(),
            # image_embeds=(b.image_embeds if with_img and fuse == "1" else torch.zeros_like(b.image_embeds)),
            # image_embeds=b.image_embeds,
            # image_attention_mask=(b.image_attention_mask if with_img else None),
            return_dict=True,
        )
        # h_tag = self.tbert(b)

        h_tit = encoded.pooler_output[0]
        h_img0 = encoded.pooler_output[1]
        
        h_tag = encoded_tag.pooler_output
        
        # h_tag = encoded_tag.pooler_output
        # h_img1 = encoded_tag.pooler_output[1]

        # h_tit = encoded.last_hidden_state[0][:, 0]
        # h_img0 = encoded.last_hidden_state[1][0][:, 0]
        
        # h_tag = encoded_tag.last_hidden_state[:, 0]
        # h_img1 = encoded_tag.last_hidden_state[1][0][:, 0]

        # emb, (emb_il, emb_if) = encoded.last_hidden_state
        # emb_tag, (emb_itl, _) = encoded_tag.last_hidden_state

        # bs = emb.size(0)
        # bs = h_tit.size(0)
        # device = emb.device

        # if not with_img:
        #     h_img = torch.zeros((bs, self.d_hidden), device=device)
        # else:
        #     h_img = emb_itl[:, 0]
        #     # h_img = encoded.pooler_output[1]
        #     h_img = self.ipooler(h_img)

        # M1
        # h_tit = emb[:, 0]
        # h_imgl = emb_il[:, 0]
        # h_imgf = emb_if[:, 0]
        # h_imgf = emb_if[:, 0]
        # h_tag = emb_tag[:, 0]

        h_tit = self.tpooler(h_tit)
        h_img0= self.ipooler(h_img0)
        h_tag = self.tpooler(h_tag)
        # h_img1= self.ipooler(h_img1)

        # h_vis = self.vpooler(h_tit + h_imgl)# + h_imgf)
        if fuse == "1":
            h_vis = self.vpooler(h_tit + h_img0)
            h_txt = self.vpooler(h_tag)
            # import pdb; pdb.set_trace()
        elif fuse == "2":
            h_vis = self.vpooler(h_tit + h_img0)
            # h_vis = F.dropout(h_vis)
            h_txt = F.dropout(h_tag)
        elif fuse == "3":
            h_vis = F.dropout(h_tit)
            h_txt = F.dropout(h_tag)

        return h_vis, h_txt
        
        # f_t = self.tmatcher(torch.cat([h_tag, h_tit, h_tag.sub(h_tit).abs()], dim=-1))
        # f_i = self.imatcher(torch.cat([h_tag, h_img, h_tag.sub(h_img).abs()], dim=-1))
        
        # y = self.fuse(torch.cat([h_tag, h_tit, h_img0], dim=-1))
        # y = self.fuse(torch.cat([f_t, f_i], dim=-1))
        # return y
        # import pdb; pdb.set_trace()

        
        # M2
        # h_tit = torch.zeros((bs, self.d_hidden), device=device)
        # h_tag = torch.zeros((bs, self.d_hidden), device=device)
        # for i in range(bs):
        #     h_tit[i] = emb[i, :b.text_length[i]].mean(dim=0)
        #     h_tag[i] = emb_tag[i, :b.tag_length[i]].mean(dim=0)

        # h_vis = self.vpooler(h_tit + h_img)

        # device = emb.device
        # import pdb; pdb.set_trace()
        
        # M1: h_tag + h_tit + h_img
        # M2: torch.cat([h_tag, h_tit, h_img], dim=-1)
        # y = self.matcher(torch.cat([h_tag, h_tit, h_img], dim=-1))
        # y = self.out(h_vis)
        # return y #h_vis, h_tag

    
    def loss_match(self, y, target):
        # loss = F.binary_cross_entropy_with_logits(y.view(-1), target.float().view(-1))
        # return [loss]

        h_vis, h_tag = y
        
        h_dist = F.cosine_similarity(h_vis, h_tag)
        h_dist[h_dist < 0] = 0.
        # loss = F.binary_cross_entropy_with_logits(y.view(-1), target.float().view(-1))
        # return [loss]
        loss = F.mse_loss(h_dist, target.float())
        # # h_seq = self.seq_relationship(h_tag)
        # # loss_nsp = F.cross_entropy(h_seq, target)
        return [loss]

        # target = target * 2 - 1
        # loss = F.cosine_embedding_loss(h_vis, h_tag, target, margin=0.1)    #
        # return [loss]
    
    def forward(self, b, with_img=True, fuse="1", rd=False):
        if not rd:
            result = self.match(b, with_img=with_img, fuse=fuse)
            loss = self.loss_match(result, b.tag_label)
        else:
            result = self.match_rd(b, with_img=with_img)
            loss = self.loss_match_rd(result, b.tag_label)
        return loss, result
    
    def match_rd(self, b, with_img=True):
        y1 = self.match(b, with_img)
        y2 = self.match(b, with_img)
        return y1, y2
    
    def loss_match_rd(self, y, target):
        y1, y2 = y
        l1 = self.loss_match(y1, target)
        l2 = self.loss_match(y2, target)
        d1 = F.cosine_similarity(*y1)
        d2 = F.cosine_similarity(*y2)
        d1[d1<=0.] = -10000.
        d2[d2<=0.] = -10000.
        l3 = torch.abs(F.kl_div(d1.view(-1).log_softmax(-1), d2.view(-1).softmax(-1), reduction="sum")) + \
                       torch.abs(F.kl_div(d2.view(-1).log_softmax(-1), d1.view(-1).softmax(-1), reduction="sum"))
        return [l1[0], l2[0], l3]