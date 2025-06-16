#!/bin/bash
#python main.py --distributed=0 --lr3=0.0001
python main.py --distributed=0 --lr3=0.003 --wd=0.1
python main.py --distributed=0 --lr3=0.003 --wd=0.2
python main.py --distributed=0 --lr3=0.003 --wd=0.5
python main.py --distributed=0 --lr3=0.003 --wd=0.05
python main.py --distributed=0 --lr3=0.002 --wd=0.1
python main.py --distributed=0 --lr3=0.005 --wd=0.1