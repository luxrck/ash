#!/bin/bash

#python -m torch.distributed.launch \
#    --nproc_per_node=2 \
#    run_pretrain.py
export with_tqdm=1

cd /data/src/vbert/

#pip install h5py
# pip install transformers
# pip install tensorboard

    # --rdzv_endpoint=0.0.0.0:12325 \
# python run_pretrain.py

python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_id=9540 \
    --rdzv_backend=c10d \
    $1 $2 $3 $4
    # run_cleandata.py #> train.log 2>&1
