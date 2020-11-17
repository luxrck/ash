import os
import re
import io
import random
import itertools
import collections
from functools import partial

from typing import Iterable

import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

from torchtext.vocab import Vocab, Vectors
from torchtext.data import Field, TabularDataset, BucketIterator, Example, Dataset

from transformers import AutoTokenizer

# import nltk
import jieba

Vocab.UNK = "[UNK]"


def zh_tokenize(s):
    # 把句子按字分开，中文按字分，英文按单词，数字按空格
    re_w = re.compile(r'[\W]+')    # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    re_zh = re.compile(r"([\u4e00-\u9fa5])")    #  [\u4e00-\u9fa5]中文范围

    partitions = re_w.split(s.lower())
    out = []
    
    for ps in partitions:
        if re_w.split(ps) == None:
            out += [t.strip() for t in jieba.cut(ps) if t]
        else:
            ret = re_zh.split(ps)
            for ch in ret:
                out.append(ch)

    out = [w for w in out if len(w.strip()) > 0]  # 去掉为空的字符

    return  out


def encode(inputs, tokenizer=None):
    assert tokenizer is not None

    out = []

    if not inputs:
        return out

    if isinstance(inputs, str):
        out = tokenizer(inputs)['input_ids']
    elif isinstance(inputs[0], str):
        out = tokenizer(inputs)['input_ids']
    elif isinstance(inputs[0], (list, tuple)):
        if isinstance(inputs[0][0], str):
            for txts in inputs:
                txt_encoded = tokenizer(txts)['input_ids']
                out.append(txt_encoded)
    return {'input_ids': out}


def chain(iterable):
    if isinstance(iterable, (str, Example)):
        yield iterable
    else:
        for item in iterable:
            yield from chain(item)


# [bs, max_seq]
# [bs, max_s1, max_s2]
def pad(data, with_length=True, device="cpu"):
    padding = 0
    length = []
    if isinstance(data[0][0], (int, float)):
        md0 = len(data)
        md1 = max([len(d1) for d1 in data])
        for d1 in range(md0):
            ld1 = len(data[d1])
            data[d1] += [0] * (md1 - ld1)
            length.append(ld1)
        data = torch.tensor(data, device=device)
        if with_length:
            return data, length
        return data
    if isinstance(data[0][0][0], (int, float)):
        md0 = len(data)
        md1 = max([len(d1) for d1 in data])
        md2 = max([max([len(d2) for d2 in d1]) for d1 in data])
        for d1 in range(md0):
            ld1 = len(data[d1])
            length_d2 = []
            for d2 in range(ld1):
                ld2 = len(data[d1][d2])
                data[d1][d2] += [0] * (md2 - ld2)
                length_d2.append(ld2)
            padding = [0] * md2
            data[d1] += [padding] * (md1 - ld1)
            length_d2 += [0] * (md1 - ld1)
            length.append(length_d2)
        data = torch.tensor(data, device=device)
        if with_length:
            return data, length
        return data
    raise ValueError("data shape must in [bs, m_seq] or [bs, m_sq1, m_sq2]")


class Batch(object):
    def __init__(self, examples, tokenizer, fields, device="cpu"):
        self._examples = examples
        self.tokenizer = tokenizer
        self._fields = fields
        self.device = device
        # self.build()
    def __iter__(self):
        return iter(self._examples)
    def __len__(self):
        return len(self._examples)
    def __getitem__(self, idx):
        return self._examples[idx]
    def build(self, tensor=False):
        for name in self._fields:
            f_val = getattr(self._examples[0], name)
            f_type = type(f_val)
            if isinstance(f_val, (list, tuple)):
                if not f_val:
                    continue
                if not isinstance(f_val[0], str):
                    continue
            f_vals = [getattr(ex, name) for ex in self._examples]
            # import pdb; pdb.set_trace()
            f_encodes = self.tokenizer(f_vals)
            f_input_ids = f_encodes['input_ids']
            if tensor:
                f_input_ids, f_lengths = pad(f_input_ids, with_length=True, device=self.device)
                setattr(self, f"{name}_length", f_lengths)
            setattr(self, name, f_input_ids)



class CharTokenizer(object):
    
    def __init__(self, vocab=None):
        self.vocab = vocab

    def build_vocab(self, dataset, fields, shared=True, min_freq=1, vectors=None, unk_init=None):
        if self.vocab is not None:
            return
        if shared:
            dataset = chain(dataset.values())
            vocab = self.build_vocab_(dataset, fields, min_freq=min_freq, vectors=vectors, unk_init=unk_init)
            self.vocab = vocab
        else:
            self.vocab = {}
            for tag, data in dataset.items():
                vocab = self.build_vocab_(data, fields, min_freq=min_freq, vectors=vectors, unk_init=unk_init)
                self.vocab[tag] = vocab

    # counter, max_size=None, min_freq=1, specials=['<pad>'], vectors=None, unk_init=None, vectors_cache=None, specials_first=True
    def build_vocab_(self, dataset, fields, min_freq=1, vectors=None, unk_init=None):
        if self.vocab:
            return self

        vocabs = []

        for name in fields:
            for example in dataset:
                attr = getattr(example, name)
                txt_tokenized = []
                # import pdb; pdb.set_trace()
                if isinstance(attr, str):
                    vocabs += self.__call__(attr, numerical=False)
                elif isinstance(attr, (list, tuple)):
                    if not attr or not isinstance(attr[0], str):
                        continue
                    txt_tokenized = self.__call__(attr, numerical=False)
                    vocabs += list(chain(txt_tokenized))
                else:
                    continue
        
        specials = ["[PAD]"] + [f"[unused{i}]" for i in range(0, 99)] + ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[unused99]"]
        vocab = Vocab(collections.Counter(vocabs),
                      min_freq=min_freq,
                      specials=specials,
                      vectors=vectors,
                      unk_init=unk_init,
                      specials_first=True)
        self.vocab = vocab
        return vocab

    def tokenize(self, inputs):
        out = zh_tokenize(inputs)
        return out

    def __call__(self, inputs, numerical=True):
        out = []

        if not inputs:
            return []

        if isinstance(inputs, str):
            # out = [t.strip() for t in jieba.cut(inputs) if t.strip()]
            out = self.tokenize(inputs)
            if numerical:
                out = [self.vocab[t] for t in out]
            # out = nltk.word_tokenize(inputs)
            # out = [t.strip() for t in inputs.lower().strip().split()]
        elif isinstance(inputs[0], str):
            for txt in inputs:
                txt_encoded = self.tokenize(txt)
                # txt_encoded = [t.strip() for t in jieba.cut(txt) if t.strip()]
                # txt_encoded = nltk.word_tokenize(txt)
                # txt_encoded = [t.strip() for t in txt.lower().strip().split()]
                if numerical:
                    txt_encoded = [self.vocab[t] for t in txt_encoded]
                out.append(txt_encoded)
        elif isinstance(inputs[0], (list, tuple)):
            if isinstance(inputs[0][0], str):
                for txts in inputs:
                    # txt_encoded = zh_tokenize(txt)
                    txt_encoded = self.__call__(txts, numerical=numerical)
                    if numerical:
                        txt_encoded = txt_encoded['input_ids']
                    out.append(txt_encoded)
        
        if numerical:
            out = {
                "input_ids": out,
                }
        
        return out



class EnglishTokenizer(CharTokenizer):
    def tokenize(self, inputs):
        inputs = re.sub(r"\.", r" \. ", inputs)
        inputs = re.sub(r"\_", r" \_ ", inputs)
        inputs = re.sub(r"\$", r" \$ ", inputs)
        out = [o.strip() for o in inputs.split() if o.strip()]
        return out



class Vocabulary(object):
    def __init__(self):
        pass



def load_for_pretrained(config):
    fields_ = {}

    if type(config['fields']) == list:
        config['fields'] = {key: (key, {}) for key in config['fields']}

    for key, (name, fargs) in config['fields'].items():
        f = Field(sequential=False, tokenize=None, use_vocab=False)
        fields_[key] = (name, f)

    shuffle = config.get("shuffle", False)
    dataset = {tag: TabularDataset(path=filename, format="json", fields=fields_) for tag, filename in config.get("dataset", {}).items()}
    
    fields_ = {name for _, (name, _) in fields_.items()}

    if "for" in config:
        pretrained_for = config["for"]
        tokenizer = AutoTokenizer.from_pretrained(pretrained_for)
        specials = [f"[unused{i}]" for i in range(0, 100)]
        tokenizer.add_special_tokens({ "additional_special_tokens": specials })
        tokenizer_ = partial(tokenizer, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)
        encoder = partial(encode, tokenizer=tokenizer_)
    else:
        tokenizer = config.get("tokenizer", CharTokenizer())
        vocab_config = config.get("vocab", {})
        if isinstance(vocab_config, (list, tuple)):
            fields_ = list(vocab_config)
            vectors = None
            share_vocab = True
        else:
            fields_ = vocab_config.get("fields", fields_)
            vectors = vocab_config.get("vectors", None)
            share_vocab = vocab_config.get("share", True)
        # import pdb; pdb.set_trace()
        tokenizer.build_vocab(dataset, fields_, shared=share_vocab, vectors=vectors, unk_init=torch.Tensor.normal_)
        encoder = tokenizer

    return {
        "vocab": tokenizer.vocab,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "iterator": {
            tag: data.DataLoader(dataset[tag].examples,
                                 batch_size=config["bs"],
                                 shuffle=shuffle,
                                 collate_fn=lambda b: Batch(b, tokenizer=encoder, fields=fields_, device=config.get("device", "cpu")),
                                 )
                        for tag in dataset
            },
        "config": config,
        }


if __name__ == "__main__":
    data = load("data/sql2nl/dev.jsonl", {
        "for": "pretrained",
        "bs": 16,
        "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "fields": ["sql", "text", "cols", "labels"],
        "io": {
            "qs-match": ("[CLS] {{ text }} [SEP] {{ sql }} [SEP]", "labels.qs"),
            "qc-match": ("[CLS] {{ text }} [SEP] {{ cols }} [SEP]", "labels.qc"),
        }
    })

    for d in data["iterator"]:
        d.build()
        import pdb; pdb.set_trace()
#add_special_tokens=False, return_token_type_ids=False, return_attention_masks=False)