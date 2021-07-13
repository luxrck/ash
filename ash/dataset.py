import os
import pdb
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

from torchtext.vocab import Vocab
from torchtext.data import Field, TabularDataset

from transformers import AutoTokenizer, BertTokenizer

# import nltk
import jieba

Vocab.UNK = "[UNK]"


def zh_char_tokenize(seq):
    results = []
    
    if isinstance(seq, str):
        seq = [seq]
    
    for s in seq:
        # 把句子按字分开，中文按字分，英文按单词，数字按空格
        # re_w = re.compile(r'[\W]+')    # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
        re_w = re.compile(r'[\s]+')    # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
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

        results.append(out)
    return {'tok/fine': results}


def zh_word_tokenize(s, with_np=True):
    import hanlp
    # HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
    
    global HanLP
    if not 'HanLP' in globals():
        HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
        # HanLP = hanlp.load(hanlp.pretrained.tok.LARGE_ALBERT_BASE)
        HanLP['tok/fine'].dict_force = {
            '剧中': ['剧', '中'],
            '其分': ['其', '分'],
            '1时': ['1', '时'],
            '2时': ['2', '时'],
            '3时': ['3', '时'],
            '4时': ['4', '时'],
            '5时': ['5', '时'],
            '6时': ['6', '时'],
            '7时': ['7', '时'],
            '8时': ['8', '时'],
            '9时': ['9', '时'],
            }
        HanLP['tok/fine'].dict_combine = {
            '哪个', '哪届', '哪次', '哪场', '哪些', '哪支', '哪条',
            '每个', '每届', '每次', '每场', '每支', '每条',
            '最小', '最大', '最少', '最多', '最低', '最高',
            '一星', '二星', '三星', '四星', '五星',
            '给出', '时长', '音乐剧', '省', '市', '县', '区', '镇', '乡', '村',
            }
    
    if isinstance(s, str):
        s = [s]
    resp = HanLP(s)
    bs = len(s)
    resp['bs'] = bs

    tokens = []
    for toks in resp['tok/fine']:
        toks_ = []
        for tok in toks:
            res = re.split("(\d+\.\d+|\d+)", tok)
            toks_ += [it for it in res if it]
        tokens.append(toks_)
    
    resp['tok/word'] = tokens

    return resp

    s_tokenize = resp['tok/fine']
    s_postag = resp['pos/ctb']
    
    s_result = []

    s_tack = []


    #
    # 方法3
    #
    def extract_pos_chunk(s):
        patterns = {
            # "p_np_1" : r"(<JJ>|<NN>|<NR>)(<NN>|<NR>|<JJ>|<DEC>|<DEG>|<VV>)*<NN>",
            # "p_np_2" : r"(<JJ>)?(<NN>|<JJ>|<DEC>|<DEG>|<VV>)*<NN>",
            "p_np_3" : r"(<JJ>|<NN>|<NR>)(<NN>|<NR>|<JJ>|<DEG>|<VV>)*<NN>",
            "p_np_2" : r"(<VV>)(<VV>|<DEC>|<NN>)*<NN>",
            "p_vp_1" : r"(<AD>)?(<VV>|<AD>|<DER>|<DEV>)*<VV>",
            "p_ot_1" : r"<[A-Z0-9]+>",
        }

        for _, p in patterns.items():
            if (r := re.match(p, s)) is not None:
                return r.group()
        
        raise Exception(f"Pos chunk not found. {s}")

    if with_np:
        for b in range(bs):
            result = []
            s_pos = ''.join([f"<{pos}>" for pos in s_postag[b]])

            i = 0
            while i < len(s_tokenize[b]):
                p_chunk = extract_pos_chunk(s_pos)
                c_size = p_chunk.count("<")
                s_chunk = ''.join(s_tokenize[b][i:i+c_size])
                result.append(s_chunk)
                s_pos = s_pos[len(p_chunk):]
                i += c_size
            s_result.append(result)
        
        resp['tok/word'] = s_result
    else:
        resp['tok/word'] = resp['tok/fine']
    return resp


    #
    # 方法2
    #
    # valid productions:
    # 1. <JJ>?(<NN>|<JJ>|<DEC>)*<NN>
    # 2. <AD>(<VA>|<VV>|<AD>)*(<VA>|<VV>)
    # 3. <VP><DEC><NN>
    i = 0
    s_tate = ""
    need_fuse = False
    while i < len(s_tokenize):
        t = s_postag[i]
        
        if s_tate == "NP":
            if t in ("JJ", "NN"):
                s_tack.append(t)
            else:
                need_fuse = True
        elif s_tate == "VP":
            if t in ("AD", "VA", "VV"):
                s_tack.append(t)
            elif t in ("DEC", "NN"):
                s_tack.append(t)
            else:
                need_fuse = True
        else:
            if t in ("JJ", "NN"):
                s_tate = "NP"
                s_tack.append(t)
            elif t in ("AD", "VA", "VV"):
                s_tate = "VP"
                s_tack.append(t)
            else:
                s_result.append(s_tokenize[i])

        if need_fuse:
            need_fuse = False
            ls = len(s_tack)
            s_result.append("".join(s_tokenize[i-ls:i]))
            s_tate = ""
            s_tack = []
            continue
        
        i += 1
    
    if s_tack:
        ls = len(s_tack)
        s_result.append("".join(s_tokenize[i-ls:i]))
        s_tate = ""
        s_tack = []


    #
    # 方法1
    #
    def traverse_tree(node):
        fused = False
        
        label = node.label()

        if label in ("CC", "CD"):
            fused = True

        if isinstance(node[0], str):
            return [node[0]], int(fused)

        chunk = []

        for subn in node:
            r, r_fused = traverse_tree(subn)
            chunk += r

            if r_fused:
                fused = True
        
        #
        # 例外情况处理
        #
        clabell = [node[i].label() for i in range(len(node))]
        clabels = " ".join(clabell)
        if label == "NP":
            if clabell[-2:] in [["JJ", "NP"], ["ADJP", "NP"]]:
                chunk = chunk[:-2] + ["".join(chunk[-2:])]
                fused = True

            # A和B
            if re.match("^(NN){1,4} CC (NN){1,4}$", clabels):
                clb = clabels.split()
                idx = clb.index("CC")
                chunk = ["".join(chunk[:idx]), chunk[idx], "".join(chunk[idx+1:])]
                fused = True

        if not fused:
            if node.label() in ("NP", "VP"):
                return ["".join(chunk)], 1
        
        return chunk, fused
    
    # s_result, _ = traverse_tree(s_postree)

    resp['tok/word'] = s_result
    return resp


def extract_np(s):
    s_tokenize = s['tok/fine']
    s_postag = s['pos/ctb']

    bs = s['bs']
    s_result = []
    
    s_tack = []


    #
    # 方法3
    #
    def extract_pos_chunk(s):
        patterns = {
            # "p_np_1" : r"(<JJ>|<NN>|<NR>)(<NN>|<NR>|<JJ>|<DEC>|<DEG>|<VV>)*<NN>",
            # "p_np_2" : r"(<JJ>)?(<NN>|<JJ>|<DEC>|<DEG>|<VV>)*<NN>",
            "p_np_3" : r"(<JJ>|<NN>|<NR>)(<NN>|<NR>|<JJ>|<DEG>|<VV>)*<NN>",
            "p_np_2" : r"(<VV>)(<VV>|<DEC>|<NN>)*<NN>",
            "p_vp_1" : r"(<AD>)?(<VV>|<AD>|<DER>|<DEV>)*<VV>",
            "p_ot_1" : r"<[A-Z0-9]+>",
        }

        for _, p in patterns.items():
            if (r := re.match(p, s)) is not None:
                return r.group()
        
        raise Exception(f"Pos chunk not found. {s}")

    for b in range(bs):
        result = []
        s_pos = ''.join([f"<{pos}>" for pos in s_postag[b]])

        i = 0
        while i < len(s_tokenize[b]):
            p_chunk = extract_pos_chunk(s_pos)
            c_size = p_chunk.count("<")
            s_chunk = ''.join(s_tokenize[b][i:i+c_size])
            result.append(s_chunk)
            s_pos = s_pos[len(p_chunk):]
            i += c_size
        s_result.append(result)
    
    s['tok/word'] = s_result

    return s



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


def isinstanceof(x, types):
    if isinstance(x, torch.Tensor):
        return isinstance(x.item(), types)
    return isinstance(x, types)



# [bs, max_seq]
# [bs, max_s1, max_s2]
def pad_(data, padding=0, with_length=True, device="cpu"):
    length = []
    if isinstanceof(data[0], (int, float)):
        data = torch.tensor(data, device=device)
        if with_length:
            return data, data.size(0)
        return data
    if isinstanceof(data[0][0], (torch.Tensor)):
        data = [item.tolist() for item in data]
    if isinstanceof(data[0][0], (int, float)):
        md0 = len(data)
        md1 = max([len(d1) for d1 in data])
        for d0 in range(md0):
            ld1 = len(data[d0])
            data[d0] += [padding] * (md1 - ld1)
            length.append(ld1)
        # pdb.set_trace()
        data = torch.tensor(data, device=device)
        if with_length:
            return data, length
        return data
    if isinstanceof(data[0][0][0], (int, float)):
        md0 = len(data)
        md1 = max([len(d1) for d1 in data])
        md2 = max([max([len(d2) for d2 in d1]) for d1 in data])
        for d0 in range(md0):
            ld1 = len(data[d0])
            length_d2 = []
            for d1 in range(ld1):
                ld2 = len(data[d1][d1])
                data[d0][d1] += [0] * (md2 - ld2)
                length_d2.append(ld2)
            padding_list = [padding] * md2
            data[d0] += [padding_list] * (md1 - ld1)
            length_d2 += [0] * (md1 - ld1)
            length.append(length_d2)
        data = torch.tensor(data, device=device)
        if with_length:
            return data, length
        return data
    raise ValueError("data shape must in [bs, m_seq] or [bs, m_sq1, m_sq2]")


def pad(data, padding=0, device="cpu"):
    return pad_(data, padding=padding, with_length=False, device=device)


def pad_with_length(data, padding=0, device="cpu"):
    return pad_(data, padding=padding, with_length=True, device=device)


def mask(lengths, device="cpu"):
    bs = len(lengths)
    max_length = max(lengths)
    return (torch.arange(max_length).expand(bs, max_length) < torch.tensor(lengths).view(bs, -1).expand(bs, max_length)).type(torch.long).to(device)


class Example(object):
    def __init__(self, ex):
        for k,v in ex.__dict__.items():
            setattr(self, k, v)

    def __getitem__(self, k):
        return self.__dict__[k]
    
    def __setitem__(self, k, v):
        setattr(self, k, v)
    
    def __contains__(self, k):
        return k in self.__dict__
    
    def get(self, k, default=None):
        return self.__dict__.get(k, default)


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
        # tokenizer = AutoTokenizer.from_pretrained(pretrained_for)
        tokenizer = BertTokenizer.from_pretrained(pretrained_for)
        specials = [f"[unused{i}]" for i in range(0, 100)]
        tokenizer.add_special_tokens({ "additional_special_tokens": specials })
        tokenizer_ = partial(tokenizer, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)
        encoder = partial(encode, tokenizer=tokenizer_)
    else:
        # TODO: vocab文件需要预处理得到，而不是使用Tokenizer在线分词得出
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
    # import pdb; pdb.set_trace()
    return {
        "vocab": tokenizer.vocab,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "iterator": {
            tag: data.DataLoader([Example(ex) for ex in dataset[tag].examples],
                                 batch_size=(config["bs"] if tag not in ("test", "dev") else 1),
                                 shuffle=(shuffle if tag not in ("test", "dev") else False),
                                 collate_fn=lambda b: Batch(b, tokenizer=encoder, fields=fields_, device=config.get("device", "cpu")),
                                 )
                        for tag in dataset
            },
        "config": config,
        }


if __name__ == "__main__":
    import sys
    qs = zh_word_tokenize(sys.argv[1])
    print(qs)
#     data = load("data/sql2nl/dev.jsonl", {
#         "for": "pretrained",
#         "bs": 16,
#         "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
#         "fields": ["sql", "text", "cols", "labels"],
#         "io": {
#             "qs-match": ("[CLS] {{ text }} [SEP] {{ sql }} [SEP]", "labels.qs"),
#             "qc-match": ("[CLS] {{ text }} [SEP] {{ cols }} [SEP]", "labels.qc"),
#         }
#     })

#     for d in data["iterator"]:
#         d.build()
#         import pdb; pdb.set_trace()
# #add_special_tokens=False, return_token_type_ids=False, return_attention_masks=False)