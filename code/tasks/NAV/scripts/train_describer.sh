#!/bin/bash

python3 -u train_describer.py -config configs/describer.yaml \
                              -executor.model.load_from experiments/nav_executor/best_val.ckpt
