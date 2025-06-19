#!/bin/bash
#python main.py --distributed=0 --lr3=0.0001
python main.py --distributed=0 --lr3=0.0002
python main.py --distributed=0 --lr3=0.0005
python main.py --distributed=0 --lr3=0.001
python main.py --distributed=0 --lr3=0.0002 --wd=0.1 --scheduler_type=null
python main.py --distributed=0 --lr3=0.0001 --wd=0.1 --scheduler_type=null