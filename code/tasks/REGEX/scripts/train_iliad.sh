#!/bin/bash

python3 -u train_student.py -config configs/iliad.yaml \
                            -executor.model.load_from experiments/regex_executor/best_dev.ckpt \
                            -describer.model.load_from experiments/regex_describer/best_dev.ckpt \
                            -student.model.load_from experiments/regex_student_unsupervised/last.ckpt
