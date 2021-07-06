#!/bin/bash

python3 -u train_student.py -config configs/iliad.yaml \
                            -executor.model.load_from experiments/nav_executor/best_val.ckpt \
                            -describer.model.load_from experiments/nav_describer/best_val.ckpt
