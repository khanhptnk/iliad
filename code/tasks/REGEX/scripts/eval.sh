#!/bin/bash

method=$1

python3 -u evaluate_student.py -config configs/${method}.yaml \
                               -student.model.load_from experiments/regex_student_${method}/best_dev.ckpt
