#!/bin/bash
#python main.py --distributed=0 --lr3=0.0001
python main.py --distributed=0 --lr3=0.002 --wd=0.1
python main.py --distributed=0 --lr3=0.002 --wd=0.1 --scheduler_type=warmup
python main.py --distributed=0 --lr3=0.002 --wd=0.1 --scheduler_type=cosine