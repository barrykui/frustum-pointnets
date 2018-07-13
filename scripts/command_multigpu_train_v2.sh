#!/bin/bash
n=8
name=mgpu_fpointnetv2
python -u train_multigpu.py \
        --model frustum_pointnets_v2 \
        --ngpu $n \
        --log_dir ${name} \
        --learning_rate 0.0001 \
        --num_point 1024 --max_epoch 201 --batch_size 24 --decay_step 800000 --decay_rate 0.5
