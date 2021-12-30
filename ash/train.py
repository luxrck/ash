import os
import re
import sys
import functools
import inspect
import random
import argparse
from collections import defaultdict, deque
from typing import Iterable

import torch
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
import torch.nn.functional as F

import transformers

import numpy as np

from tqdm import tqdm



# FIXME: 注意！DDP会调用model.forward, 因此不能自己自定义forward函数。
# 如果需要自定义forward函数，务必使用Mix.fwd进行wrap.
class Mix(object):
    def __init__(self, instance):
        super().__init__()
        self.instance = instance
        self.submodule = getattr(self.instance, 'module', self.instance)
        self.submodule_forward = self.submodule.forward
        # self.forward_map = {}

    # @fwd
    # def fwd_func(self, *args, **kwargs):    # self: ddp.module
    #     ...
    # fwd_func()
    #     -> Mix.fwd:inner_forward
    #         -> ddp.module.forward = fwd_func
    #         -> ddp.forward
    def fwd(self, forward_func):
        def forward_wrapper():
            def inner_forward(*args, **kwargs):
                self.submodule.forward = forward_func.__get__(self.submodule)
                result = self.instance(*args, **kwargs)
                self.submodule.forward = self.submodule_forward
                return result
            # self.forward_map[forward_func.__name__] = inner_forward
            return inner_forward
        return forward_wrapper()

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)

    # __getattribute__ -> [fwd, __call__]
    # if failed:
    #   __getattr__ -> [submodule.attrs, instance.attrs]
    def __getattr__(self, k):
        # v = getattr(self.submodule, k, None)
        # if v:
            # return v
        return getattr(self.instance, k)


class RuntimeInfo(object):

    __nmae__ = "runtimeinfo"

    def __init__(self):
        super().__init__()


class TensorBoard(object):

    __name__ = "tensorboard"

    def __init__(self):
        super().__init__()
        self.writer = SummaryWriter(log_dir=os.environ.get('tb', None))
    
    def bind(self, app):
        self.app = app
        @app.on("iter_completed")
        def update_loss(e):
            if not e.app.main_device():
                return
            if e.current_iter % 500 != 0:
                return
            self.add_scalar_("Loss/train", e.avg_loss, e.current_iter)
            # self.add_scalar_("Loss/valid", e.avg_loss_validate, e.current_iter)
    
    def add_scalar_(self, *args, **kwargs):
        self.writer.add_scalar(*args, **kwargs)
        return self



class Arguments(object):
    
    __name__ = "cli_arguments"

    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser()
        self.init_flags_()
    
    def bind(self, app):
        self.app = app
    
    def init_flags_(self):
        self.parser.add_argument("--local_rank", type=int)
    
    def parse(self):
        args = self.parser.parse_args()
        self.app.ddp_local_rank = args.local_rank
        return self



class TQDM(object):

    __name__ = "tqdm"

    def __init__(self):
        super().__init__()
        self.progress = None
        self.use_epoch = True
        self.current_iter = 1
        self.current_epoch = 1
        self.max_iters = 1
        self.max_epochs = 1
    
    def bind(self, app):
        self.app = app
        @app.on("train_started")
        def initialize(e):
            self.initialize_(
                current_iter=e.current_iter,
                current_epoch=e.current_epoch,
                max_iters=e.max_iters,
                max_epochs=e.max_epochs,
            )
        def update(e):
            if e.name == "iter_completed":
                to = "iter"
            elif e.name == "epoch_completed":
                to = "epoch"
            else:
                return
            self.update_(to=to,
                         iter=e.current_iter,
                         epoch=e.current_epoch,
                         loss=e.loss,
                         loss_=e.loss_,
                         avg_loss=e.avg_loss)
        app.on("iter_completed", update)
        app.on("epoch_completed", update)
    
    def initialize_(self, current_iter=1, current_epoch=1, max_iters=1, max_epochs=1):
        assert max_iters >= 0 or max_epochs >= 0

        use_epoch = True if max_epochs > 0 else False

        self.use_epoch = use_epoch
        self.current_iter = current_iter
        self.current_epoch = current_epoch
        self.max_iters = max_iters
        self.max_epochs = max_epochs

        #if not self.app.main_device():
        #    return self
        
        max_iter_or_epoch = self.max_epochs if self.use_epoch else self.max_iters
        current_iter_or_epoch = self.current_epoch if self.use_epoch else self.current_iter
        self.progress = tqdm(total=max_iter_or_epoch, miniters=0)
        self.progress.n = current_iter_or_epoch
        self.progress.last_print_n = current_iter_or_epoch
        return self

    def update_(self, to="epoch", **kwargs):
        if self.progress is None:
            return
        rank = self.app.rank()
        kwargs['rank'] = rank
        if kwargs:
            if 'loss_' in kwargs:
                l_ = kwargs['loss_']
                l_ = ", ".join(["%.3f" % l for l in l_]).strip()
                kwargs['loss_'] = f"[{l_}]"
            self.progress.set_postfix(**kwargs)
        if to == "iter" and self.use_epoch:
            return
        if to == "epoch" and not self.use_epoch:
            return
        # import pdb; pdb.set_trace()
        self.progress.update(1)
        return self



class Checkpoint(object):

    __name__ = "checkpoint"

    def __init__(self, root="checkpoint"):
        self.checkpoint_root = root

    def bind(self, app):
        self.app = app
        self.fastforwarded = False
        self.checkpoint_root = os.path.join(app.app_root, self.checkpoint_root)

    def set_checkpoint_root(self, root):
        self.checkpoint_root = os.path.join(self.app.app_root, root)
        return self

    def fastforward(self, to="last", with_optim=False):
        # self._fastforward(to=to)
        @self.app.on("initialize")
        def forward(e):
            self.fastforward_(to=to, with_optim=with_optim)
        return self
    def fastforward_(self, to="last", with_optim=False):
        app = self.app
        checkpoint_root = self.checkpoint_root
        model_name = app.name()
        if not os.path.exists(checkpoint_root):
            os.mkdir(checkpoint_root)
        ckpt = None
        to_model = None
        fastforward = list(os.walk(checkpoint_root))[0][2]
        if to == "best":
            to_model = f"{model_name}.best.pt"
        elif to is None:
            pass
        else: # to == "last" or other regex.
            # import pdb; pdb.set_trace()
            # key = lambda x: re.findall(f"{model_name}.*\.it(?P<iter>\d+)\..*\.pt", x) or [-1]
            # fastforward = list(filter(key, fastforward))
            # if fastforward:
            #     fastforward = sorted(fastforward, key=lambda x: int(key(x)[0]))
            #     to_model = fastforward[-1]
            index = -1; max_iters = 0
            for i,name in enumerate(fastforward):
                real_name = re.search(r"(?P<name>[a-zA-Z0-9-_+~\:\.]+?)\.(ep|EP)\d+\..*", name)
                if not real_name:
                    real_name = re.search(r"(?P<name>[a-zA-Z0-9-_+~\:\.]+?)\.(it|IT)\d+\..*", name)
                if real_name:
                    real_name = real_name.group("name")
                if not model_name == real_name:
                    continue
                
                # print(model_name, real_name)
                
                if to != "last":
                    if to == name or re.search(to, name) is not None:
                        to_model = name
                        break
                else:
                    model_iters = re.search(r".*\.it(?P<iter>\d+)\..*", name)
                    if not model_iters:
                        continue
                    model_iters = model_iters.group("iter")
                    if not model_iters:
                        continue
                    model_iters = int(model_iters)
                    if model_iters > max_iters:
                        max_iters = model_iters
                        index = i
            if index >= 0:
                to_model = fastforward[index]
        
        if to_model:
            model_location = os.path.join(checkpoint_root, to_model)
            print("Fastforward to:", model_location)
            ckpt = torch.load(model_location, map_location="cpu")#app.device())
        
        if ckpt:
            if "start" in ckpt:
                app.current_iter = ckpt["start"] + 1
            
            if "epoch" in ckpt:
                app.current_epoch = ckpt["epoch"] + 1
            
            try:
                app.model.load_state_dict(ckpt["model"], strict=True)
            except Exception as err:
                print(err)
                app.model.load_state_dict(ckpt["model"], strict=False)
            
            if with_optim:
                if "optim" in ckpt:
                    for op in ckpt["optim"]:
                        app.optimizers[op].load_state_dict(ckpt["optim"][op])
                
                if "sched" in ckpt:
                    for op in ckpt["sched"]:
                        app.schedulers[op].load_state_dict(ckpt["sched"][op])
                #app.optimizer.load_state_dict(ckpt["optim"])
            # By default all the modules are initialized to train mode (self.training = True).
            # Also be aware that some layers have different behavior during train/and evaluation
            # (like BatchNorm, Dropout) so setting it matters.
            app.model.train()
        return self

    def save(self, root=None, with_optim=False):
        app = self.app
        checkpoint_root = root if root else self.checkpoint_root
        current_epoch = app.current_epoch
        current_iter = app.current_iter
        ckpt = {
            "start": current_iter,
            "epoch": current_epoch,
            "model": app.model.state_dict()
            }
        if with_optim:
            ckpt["optim"] = {
                op: app.optimizers[op].state_dict() for op in app.optimizers
            }
            if app.schedulers:
                ckpt["sched"] = {
                    op: app.schedulers[op].state_dict() for op in app.schedulers
                }
        torch.save(ckpt,
            os.path.join(checkpoint_root,
                            f"{app.name()}.ep{current_epoch}.it{current_iter}.pt"))

    def save_every(self, iters=-1, epochs=-1, with_optim=False):
        def save_(e, use_epoch=False):
            # print(self.app.main_device(), use_epoch, e.current_iter, iters)
            if not self.app.main_device():
                return
            flag = e.current_epoch % epochs == 0 if use_epoch else e.current_iter % iters == 0
            if flag:
                self.save(with_optim=with_optim)
        if epochs > 0:
            self.app.on("epoch_completed", functools.partial(save_, use_epoch=True))
            return self
        if iters > 0:
            self.app.on("iter_completed", functools.partial(save_, use_epoch=False))
            return self
        
        return self



class App(object):
    r'''
    Events:
        initialize

        train_started:
            epoch_started:
                iter_started
                train:
                iter_completed:
            epoch_completed / validate:
        train_completed:

        evaluate:
    '''

    class Event(object):
        def __init__(self, name, meta, **kwargs):
            self.name = name
            self.meta = meta
            self.__kw__ = {}
            self.__dict__.update(kwargs)
            self.__kw__.update(kwargs)
        def __setattr__(self, k, v):
            self.__dict__[k] = v
        def kw(self):
            return self.__kw__

    def __init__(self, model, name="", root=".", device="cpu", **kwargs):
        # self.app = app
        self.model = model
        self.model_name = model.__class__.__name__
        self.app_name = name
        self.app_root = root
        self.optimizers = {}
        self.schedulers = {}
        self.optimizer_builders = {}
        self.scheduler_builders = {}
        self.device_ = torch.device(device)
        self.config(**kwargs)

        self.use_amp = False
        self.use_ddp = False
        self.ddp_local_rank = -1

        # self.event_map = defaultdict(lambda: {"handlers": set(), "uniq": False})
        self.event_map = defaultdict(set)

        # FUCK NO !!!!!!!!
        # self.event_map["train"] = self.event_map["iter_started"]
        
        self.event_q = []

        self.extension_map = {}
        self.current_iter = 1
        self.current_epoch = 1
    
    def name(self, format=""):
        model_name = self.model_name
        app_name = self.app_name
        if not format:
            if app_name:
                return app_name
            else:
                return model_name
        return format.format(model=model_name, app=app_name)

    def config(self, **kwargs):
        default = {}
        default.update(kwargs)
        self.c = default
        for key,val in default.items():
            self.__setattr__(key, val)
        return self
    
    def emit(self, e):
        pass

    def to(self, device):
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device
        self.model = self.model.to(device)
        for op in self.optimizer_builders:
            self.optimizers[op] = self.optimizer_builders[op]()
            if self.scheduler_builders.get(op, None):
                sc = self.scheduler_builders[op](self.optimizers[op])
                self.schedulers[op] = sc
        return self
    
    def extend(self, ext):
        ext.bind(self)
        self.extension_map[ext.__name__] = ext
        return self

    def exec_handles(self, on_event, meta, **kwargs):
        # print("Exec Event:", on_event)
        e = App.Event(name=on_event, meta=meta, **kwargs)
        e.app = self
        e.model = self.model
        e.device = self.device
        e.optimizers = self.optimizers
        return list(map(lambda h: h(e), self.event_map[on_event]))
        # results = []
        # for handler in self.event_map[on_event]:
        #     hres = handler(e)
        #     results.append(hres)
        # return results

    def on(self, event, handler=None):
        def event_wrapper(handler):
            def inner_event_wrapper(*args, **kwargs):
                return handler(*args, **kwargs)
            self.event_map[event].add(inner_event_wrapper)
            return inner_event_wrapper
        if handler is not None:
            return event_wrapper(handler)
        return event_wrapper
    
    def with_seed(self, i):
        offset = self.rank()
        np.random.seed(i + offset)
        random.seed(i + offset)
        torch.manual_seed(i + offset)
        torch.cuda.manual_seed_all(i)
        return self

    def with_amp(self, flag=False, **kwargs):
        self.use_amp = flag
        if not flag:
            return self
        self.amp_scaler = amp.GradScaler(**kwargs)
        return self
    
    def with_optimizer(self, op, params=None, scheduler=None, **kwargs):
        if issubclass(op, torch.optim.Optimizer):
            if not params:
                params = [{"params": self.model.parameters()}]
        else:
            op = functools.partial(op, self)
        op_name = f"{op.__name__}:{len(self.optimizers)}"
        optimizer_builder = functools.partial(op, params, **kwargs)
        # self.optimizers[op_name] = optimizer_builder()
        self.optimizer_builders[op_name] = optimizer_builder
        if scheduler:
            self.scheduler_builders[op_name] = scheduler
        return self
    
    def with_data_parallel(self, find_unused_parameters=False):
        self.use_ddp = True
        # os.environ["OMP_NUM_THREADS"] = "1"
        # os.environ["MASTER_ADDR"] = "0.0.0.0"
        # print(os.environ["MASTER_ADDR"])
        # print(os.environ["MASTER_PORT"])
        # if master_port > 0:
            # os.environ["MASTER_PORT"] = str(master_port)
        # self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        # self.ddp_world_size = world_size
        self.ddp_find_unused_parameters = find_unused_parameters
        # self.ddp_init_method = init_method
        return self
    
    def device(self):
        lr = self.local_rank()
        if lr >= 0:
            self.device_ = torch.device(self.local_rank())
        return self.device_

    def rank(self):
        if self.use_ddp:
            return int(os.environ.get('RANK', -1))
            # return torch.distributed.get_rank()
        return 0
    
    def local_rank(self):
        return int(os.environ.get('LOCAL_RANK', -1))
    
    def world_size(self):
        return int(os.environ.get('WORLD_SIZE', -1))
    
    def main_device(self):
        return self.rank() == 0

    def build(self, device="cuda"):
        if self.use_ddp and self.world_size() > 0:
            torch.cuda.set_device(self.device())
            self.to(self.device())

            # if self.ddp_init_method == "tcp":
            #     init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            # else:
            #     init_method = "env://"

            torch.distributed.init_process_group(
                backend='nccl',
                # init_method=init_method,
                world_size=self.world_size(),
                rank=self.rank(),
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                find_unused_parameters=self.ddp_find_unused_parameters,
                device_ids=[self.local_rank()],
                output_device=self.local_rank()
            )
            # self.model = model
            # self.model = Mix(self.model)
            # M = DDP(M)
            # M() -> DDP.__call__ -> DDP.forward -> M.forward
            # r = M.module.func()
            # l = M.module.loss(r)
            # self.model = model
            # ddp_sampler = DistributedSampler(data.dataset, rank=self.rank(), drop_last=True)
            # ddp_data = DataLoader(data.dataset,
            #                       batch_size=data.batch_size,
            #                       shuffle=False,
            #                       num_workers=int(os.environ.get('num_workers', 0)),
            #                       collate_fn=data.collate_fn,
            #                       sampler=ddp_sampler)
            # data = ddp_data
        else:
            self.to(device)
        
        # self.model = Mix(self.model)
        
        self.model.train()
        self.exec_handles("initialize", meta=None)
        return self

    def eval(self, dataset, bs=1, shuffle=False, num_workers=0, collate_fn=None, seed=0):
        if seed >= 0:
            self.with_seed(seed)

        result = {
            "count": 0,
            "loss": [],
            "predict": [],
            "gold": [],
            "result": [],
        }

        # if not self.main_device():
        #     return None
        
        if dataset is None:
            return None

        if self.use_ddp and self.world_size() > 0:
        #     torch.distributed.init_process_group(
        #         backend='nccl',
        #         init_method='env://',
        #         world_size=self.world_size(),
        #         rank=self.rank(),
        #     )
        #     model = nn.parallel.DistributedDataParallel(
        #         self.model,
        #         find_unused_parameters=False,
        #         device_ids=[self.local_rank()],
        #         output_device=self.local_rank()
        #     )
        #     self.model = Mix(model, ["module"])
        #     # self.model = model
            ddp_sampler = DistributedSampler(dataset, rank=self.rank(), shuffle=False, drop_last=True)
            ddp_data = DataLoader(dataset,
                                  batch_size=bs,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  sampler=ddp_sampler)
            data = ddp_data
        else:
            data = DataLoader(dataset,
                              batch_size=bs,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              collate_fn=collate_fn)

        valid_iters = 0
        total_loss = 0
        self.model.eval()
        
        with torch.no_grad():
            # if self.use_ddp:
            #     ddp_sampler.set_epoch(current_epoch)
            for i, batch in tqdm(enumerate(data)):
                perd = None
                
                # TODO: Not a good implementation...
                with amp.autocast(self.use_amp):
                    pred = self.exec_handles("evaluate",
                                                meta=None,
                                                current_iter=i,
                                                batch=batch)[0]
                if pred is not None:
                    valid_iters += 1
                    loss = pred["loss"]
                    if isinstance(loss, torch.Tensor):
                        loss = [loss.item()]
                    total_loss += sum(loss)
                    result["count"] += 1
                    result["loss"].append(loss)
                    result["predict"].append(pred["predict"])
                    result["gold"].append(pred["gold"])
                    result["result"].append(pred)
        if valid_iters == 0:
            valid_iters = 1
        result["loss_avg"] = total_loss / valid_iters
        return result
        
    def run(self, dataset, bs=1, shuffle=True, num_workers=0, collate_fn=None, validate=None, max_iters=-1, max_epochs=-1, accumulate=1, train=True):
        if not train:
            return self

        self.model.train()

        print("WORLD_SIZE:", self.world_size())
        print("LOCAL_RANK:", self.local_rank())
        print("RANK:", self.rank())

        if self.use_ddp and self.world_size() > 0:
        #     torch.distributed.init_process_group(
        #         backend='nccl',
        #         init_method='env://',
        #         world_size=self.world_size(),
        #         rank=self.rank(),
        #     )
        #     model = nn.parallel.DistributedDataParallel(
        #         self.model,
        #         find_unused_parameters=False,
        #         device_ids=[self.local_rank()],
        #         output_device=self.local_rank()
        #     )
        #     self.model = Mix(model, ["module"])
        #     # self.model = model
            ddp_sampler = DistributedSampler(dataset, rank=self.rank(), shuffle=shuffle, drop_last=True)
            ddp_data = DataLoader(dataset,
                                  batch_size=bs,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  sampler=ddp_sampler)
            data = ddp_data
        else:
            data = DataLoader(dataset,
                              batch_size=bs,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
            ddp_sampler = None

        assert max_epochs >= 0 or max_iters >= 0
        assert accumulate > 0

        current_iter = self.current_iter
        current_epoch = self.current_epoch

        use_epoch = True if max_epochs else False
        
        meta = App.Event(name="meta", meta=None)
        self.exec_handles("train_started",
                          meta,
                          current_iter=current_iter,
                          current_epoch=current_epoch,
                          max_iters=max_iters,
                          max_epochs=max_epochs,
                          use_epoch=use_epoch)

        # self.model.zero_grad()
        for op in self.optimizers:
            self.optimizers[op].zero_grad()

        # torch.distributed.barrier()

        while (current_iter < max_iters + 1 and max_epochs <= 0) \
           or (current_epoch < max_epochs + 1):
            if self.use_ddp:
                if ddp_sampler is not None:
                    ddp_sampler.set_epoch(current_epoch)
            
            iterator = enumerate(data)
            # print("Epoch: ", current_epoch)

            loss_avg = 0
            loss_avg_validate = 0
            valid_iters = 0
            # loss = 0
            loss_ = []
            loss_accumulate = 0

            loss_lasts = [0] * 100 # max size: 500 iters
            # loss_lasts_count = 0
            # loss_lasts_sum = 0

            self.exec_handles("epoch_started",
                              meta,
                              current_epoch=current_epoch,
                              current_iter=current_iter,
                              max_iters=max_iters,
                              max_epochs=max_epochs,
                              use_epoch=use_epoch)

            for i,batch in iterator:
                for op in self.optimizers:
                    self.optimizers[op].zero_grad()

                self.current_epoch = current_epoch
                self.current_iter = current_iter

                self.exec_handles("iter_started",
                                  meta,
                                  current_epoch=current_epoch,
                                  current_iter=current_iter,
                                  max_iters=max_iters,
                                  max_epochs=max_epochs,
                                  batch=batch,
                                  i=i)
                
                def model_forward():
                    with amp.autocast(self.use_amp):
                        loss = self.exec_handles("train",
                                                meta,
                                                current_epoch=current_epoch,
                                                current_iter=current_iter,
                                                max_iters=max_iters,
                                                max_epochs=max_epochs,
                                                batch=batch,
                                                i=i)[0]

                        if loss is None:
                            return None

                        loss_ = [loss] if isinstance(loss, (torch.Tensor, int, float)) else loss
                        
                        if loss_[0] < 0:
                            return None

                        loss = sum(loss_)
                        loss_accumulate = loss.item()
                        loss_ = [l.item() for l in loss_]

                        loss /= accumulate

                    return {
                        "loss": loss,
                        "loss_accumulate": loss_accumulate,
                        "loss_": loss_
                    }

                if (valid_iters + 1) % accumulate != 0 and self.use_ddp and self.world_size() > 0:
                    with self.model.no_sync():
                        loss = model_forward()
                else:
                    loss = model_forward()

                if not loss:
                    continue

                if self.use_amp:
                    self.amp_scaler.scale(loss["loss"]).backward()
                else:
                    loss["loss"].backward()
                
                # loss_accumulate += sum(loss_)

                # len(dataset) 不一定被bs整除
                if (valid_iters + 1) % accumulate == 0:
                    # loss_accumulate /= accumulate
                    
                    if self.use_amp:
                        # self.amp_scaler.scale(loss_accumulate).backward()
                        # import pdb; pdb.set_trace()
                        # for op in self.optimizers:
                            # self.amp_scaler.unscale_(self.optimizers[op])

                        # # try:
                        #     # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2, error_if_nonfinite=False)
                        # # except Exception as err:
                        #     # import pdb; pdb.set_trace()
                        
                        for op in self.optimizers:
                            self.amp_scaler.step(self.optimizers[op])
                        
                        self.amp_scaler.update()
                    else:
                        # import pdb; pdb.set_trace()
                        # loss_accumulate.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2, error_if_nonfinite=False)
                    
                        for op in self.optimizers:
                            self.optimizers[op].step()
                    
                    for op in self.schedulers:
                        self.schedulers[op].step()

                    # self.model.zero_grad()
                    # for op in self.optimizers:
                        # self.optimizers[op].zero_grad()

                    # loss = loss.item()
                    # avg_loss += loss_accumulate
                    
                valid_iters += 1
                
                loss_accumulate = loss["loss_accumulate"]
                loss_ = loss["loss_"]
                
                loss_avg += loss_accumulate

                # 只计算最后K个iters的avg loss
                loss_avg -= loss_lasts.pop(0)
                loss_lasts.append(loss_accumulate)
                
                self.exec_handles("iter_completed",
                                  meta,
                                  data=validate,
                                  current_epoch=current_epoch,
                                  current_iter=current_iter,
                                  max_iters=max_iters,
                                  max_epochs=max_epochs,
                                  loss=loss_accumulate,
                                  loss_=loss_,
                                  avg_loss=loss_avg / 100,
                                  batch=batch,
                                  i=i)
                
                if (valid_iters + 1) % accumulate == 0:
                    loss_accumulate = 0.
                
                current_iter += 1
                
                if not use_epoch:
                    if current_iter >= max_iters + 1:
                        break
            
            # self.progress.update(to = "epoch")
            # self.tensorboard.add_scalar("Loss/train", avg_loss.item() / valid_iters, current_epoch)
            predict = None
            if validate is not None:
                predict = self.eval(validate)
                if predict:
                    loss_avg_validate = predict["loss"]
                self.model.train()

            self.exec_handles("epoch_completed",
                              meta,
                              predict=predict,
                              current_epoch=current_epoch,
                              current_iter=current_iter,
                              max_iters=max_iters,
                              max_epochs=max_epochs,
                              use_epoch=use_epoch,
                              loss=loss_accumulate,
                              loss_=loss_,
                              avg_loss=loss_avg / 100,
                              avg_loss_validate=loss_avg_validate)
            
            current_epoch += 1

        self.exec_handles("train_completed",
                          meta,
                          current_epoch=current_epoch,
                          current_iter=current_iter,
                          max_iters=max_iters,
                          max_epochs=max_epochs,
                          use_epoch=use_epoch)
        return self

    # Called when the default attribute access fails with an AttributeError (either __getattribute__() raises an
    # AttributeError because name is not an instance attribute or an attribute in the class tree for self; or __get__()
    # of a name property raises AttributeError). This method should either return the (computed) attribute value or raise
    # an AttributeError exception.
    # https://docs.python.org/3/reference/datamodel.html#object.__getattr__
    # Trainer包装App, 当Train没有属性k时, 可以从App中查找, 但最后要返回Trainer的Instance.
    def __getattr__(self, k):
        class Self(object):
            def __init__(self, prev_self, chained):
                self._self = prev_self
                self.chained = chained
            def __call__(self, *args, **kwargs):
                self.chained(*args, **kwargs)
                return self._self
        v = None
        for ext_name, ext in self.extension_map.items():
            try:
                if k == ext_name:
                    v = ext
                else:
                    v = ext.__getattribute__(k)
                break
            except AttributeError:
                pass
        #if not v:
        #    v = self.app.__getattribute__(k)
        # import pdb; pdb.set_trace()
        if inspect.ismethod(v):
            return Self(self, v)
        return v
