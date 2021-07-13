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



class Mix(object):
    def __init__(self, instance, submodules):
        super().__init__()
        self.instance = instance
        self.submodules = submodules
    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)
    def __getattr__(self, k):
        v = None
        for module in self.submodules:
            try:
                module = getattr(self.instance, module)
                v = getattr(module, k)
            except AttributeError:
                pass
        if not v:
           v = getattr(self.instance, k)
        return v



class TensorBoard(object):

    __name__ = "tensorboard"

    def __init__(self):
        super().__init__()
        self.writer = SummaryWriter()
    
    def bind(self, app):
        self.app = app
        @app.on("epoch_completed")
        def update_loss(e):
            self.add_scalar_("Loss/train", e.avg_loss, e.current_epoch)
            self.add_scalar_("Loss/valid", e.avg_loss_validate, e.current_epoch)
    
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

        if not self.app.main_device():
            return self
        
        max_iter_or_epoch = self.max_epochs if self.use_epoch else self.max_iters
        current_iter_or_epoch = self.current_epoch if self.use_epoch else self.current_iter
        self.progress = tqdm(total=max_iter_or_epoch, miniters=0)
        self.progress.n = current_iter_or_epoch
        self.progress.last_print_n = current_iter_or_epoch
        return self

    def update_(self, to="epoch", **kwargs):
        if self.progress is None:
            return
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

    def fastforward(self, to="last"):
        # self._fastforward(to=to)
        @self.app.on("initialize")
        def forward(e):
            self.fastforward_(to=to)
        return self
    def fastforward_(self, to="last"):
        app = self.app
        checkpoint_root = self.checkpoint_root
        model_name = app.name()
        if not os.path.exists(checkpoint_root):
            os.mkdir(checkpoint_root)
        ckpt = None
        to_model = None
        fastforward = list(os.walk(checkpoint_root))[0][2]
        if to == "last":
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
        elif to == "best":
            to_model = f"{model_name}.best.pt"
        elif to is not None:
            for name in fastforward:
                if not model_name in name:
                    continue
                if to == name or re.search(to, name) is not None:
                    to_model = name
                    break
        else:
            pass
        if to_model:
            model_location = os.path.join(checkpoint_root, to_model)
            print("Fastforward to:", model_location)
            ckpt = torch.load(model_location, map_location=app.device)
        if ckpt:
            if "start" in ckpt:
                app.current_iter = ckpt["start"] + 1
            if "epoch" in ckpt:
                app.current_epoch = ckpt["epoch"] + 1
            app.model.load_state_dict(ckpt["model"])
            if "optim" in ckpt:
                app.optimizer.load_state_dict(ckpt["optim"])
            # By default all the modules are initialized to train mode (self.training = True).
            # Also be aware that some layers have different behavior during train/and evaluation
            # (like BatchNorm, Dropout) so setting it matters.
            app.model.train()
        return self

    def save_every(self, iters=-1, epochs=-1):
        def save(e, use_epoch=False):
            if not self.app.main_device():
                return
            current_iter = e.current_iter
            current_epoch = e.current_epoch
            flag = current_epoch % epochs == 0 if use_epoch else current_iter % iters == 0
            if flag:
                ckpt = {
                    "start": current_iter,
                    "epoch": current_epoch,
                    "model": e.model.state_dict()
                    }
                torch.save(ckpt,
                    os.path.join(self.checkpoint_root,
                                 f"{e.app.name()}.ep{current_epoch}.it{current_iter}.pt"))

        if epochs > 0:
            self.app.on("epoch_completed", functools.partial(save, use_epoch=True))
            return self
        if iters > 0:
            self.app.on("iter_completed", functools.partial(save, use_epoch=False))
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
        self.device = device
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
        self.device = device
        self.model = self.model.to(device)
        for op in self.optimizers:
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
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)
        return self

    def with_amp(self, **kwargs):
        self.use_amp = True
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
        self.optimizers[op_name] = optimizer_builder()
        self.optimizer_builders[op_name] = optimizer_builder
        if scheduler:
            self.scheduler_builders[op_name] = scheduler
        return self
    
    def with_data_parallel(self, world_size=-1, master_port=-1):
        self.use_ddp = True
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        if master_port > 0:
            os.environ["MASTER_PORT"] = str(master_port)
        self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        self.ddp_world_size = world_size
        self.device = torch.device(self.ddp_local_rank)
        return self
    
    def rank(self):
        if self.use_ddp:
            return int(os.environ['LOCAL_RANK'])
            # return torch.distributed.get_rank()
        return 0
    
    def main_device(self):
        return self.rank() == 0

    def build(self):
        if self.use_ddp:
            torch.cuda.set_device(self.ddp_local_rank)
            self.to(self.device)
        self.model.train()
        self.exec_handles("initialize", meta=None)
        return self

    def eval(self, data=None):
        result = {
            "loss": 0.,
            "count": 0,
            "predict": [],
            "gold": [],
            "result": [],
        }

        if not self.main_device():
            return None
        
        if data is None:
            return None

        valid_iters = 0
        self.model.eval()
        
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data)):
                # TODO: Not a good implementation...
                pred = self.exec_handles("evaluate",
                                            meta=None,
                                            batch=batch)[0]
                if pred is not None:
                    valid_iters += 1
                    loss = pred["loss"]
                    if isinstance(loss, torch.Tensor):
                        loss = loss.item()
                    result["loss"] += loss
                    result["count"] += 1
                    result["predict"].append(pred["predict"])
                    result["gold"].append(pred["gold"])
                    result["result"].append(pred)
        if valid_iters == 0:
            valid_iters = 1
        result["loss"] /= valid_iters
        return result
        
    def run(self, data, validate=None, max_iters=-1, max_epochs=-1, accumulate=1, train=True):
        if not train:
            return self

        self.model.train()

        if self.use_ddp:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=self.ddp_local_rank
            )
            model = nn.parallel.DistributedDataParallel(
                self.model,
                find_unused_parameters=True,
                device_ids=[self.ddp_local_rank],
                output_device=self.ddp_local_rank
            )
            self.model = Mix(model, ["module"])
            # self.model = model
            ddp_sampler = DistributedSampler(data.dataset, rank=self.ddp_local_rank)
            ddp_data = DataLoader(data.dataset,
                                  batch_size=data.batch_size,
                                  shuffle=False,
                                  collate_fn=data.collate_fn,
                                  sampler=ddp_sampler)
            data = ddp_data

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

        while (current_iter < max_iters + 1 and max_epochs <= 0) \
           or (current_epoch < max_epochs + 1):
            iterator = enumerate(data)

            avg_loss = 0
            avg_loss_validate = 0
            valid_iters = 0
            # loss = 0
            loss_ = []
            loss_accumulate = 0

            self.exec_handles("epoch_started",
                              meta,
                              current_epoch=current_epoch,
                              current_iter=current_iter,
                              max_iters=max_iters,
                              max_epochs=max_epochs,
                              use_epoch=use_epoch)

            for i,batch in iterator:
                self.exec_handles("iter_started",
                                  meta,
                                  current_epoch=current_epoch,
                                  current_iter=current_iter,
                                  max_iters=max_iters,
                                  max_epochs=max_epochs,
                                  batch=batch,
                                  i=i)

                if self.use_amp:
                    with amp.autocast():
                        loss = self.exec_handles("train",
                                                meta,
                                                current_epoch=current_epoch,
                                                current_iter=current_iter,
                                                max_iters=max_iters,
                                                max_epochs=max_epochs,
                                                batch=batch,
                                                i=i)[0]
                else:
                    loss = self.exec_handles("train",
                                            meta,
                                            current_epoch=current_epoch,
                                            current_iter=current_iter,
                                            max_iters=max_iters,
                                            max_epochs=max_epochs,
                                            batch=batch,
                                            i=i)[0]

                if loss is None:
                    continue

                loss_ = [loss] if isinstance(loss, (torch.Tensor)) else loss
                
                if loss_[0] < 0:
                    continue

                loss = sum(loss_)
                loss /= accumulate

                if self.use_amp:
                    self.amp_scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                loss_accumulate += loss.item()
                # loss_accumulate += sum(loss_)

                if (i + 1) % accumulate == 0:
                    # loss_accumulate /= accumulate
                    
                    if self.use_amp:
                        # self.amp_scaler.scale(loss_accumulate).backward()
                        for op in self.optimizers:
                            self.amp_scaler.step(self.optimizers[op])
                        self.amp_scaler.update()
                    else:
                        # import pdb; pdb.set_trace()
                        # loss_accumulate.backward()
                        for op in self.optimizers:
                            self.optimizers[op].step()
                        for op in self.schedulers:
                            self.schedulers[op].step()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                    
                    # self.model.zero_grad()
                    for op in self.optimizers:
                        self.optimizers[op].zero_grad()

                    # loss = loss.item()
                    loss_ = [l.item() for l in loss_]
                    avg_loss += loss_accumulate
                    
                valid_iters += 1
                
                self.exec_handles("iter_completed",
                                  meta,
                                  data=validate,
                                  current_epoch=current_epoch,
                                  current_iter=current_iter,
                                  max_iters=max_iters,
                                  max_epochs=max_epochs,
                                  loss=loss_accumulate,
                                  loss_=loss_,
                                  avg_loss=avg_loss * accumulate / valid_iters,
                                  batch=batch,
                                  i=i)
                
                if (i + 1) % accumulate == 0:
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
                    avg_loss_validate = predict["loss"]
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
                              avg_loss=avg_loss * accumulate / valid_iters,
                              avg_loss_validate=avg_loss_validate)
            
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
        for _,ext in self.extension_map.items():
            try:
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
