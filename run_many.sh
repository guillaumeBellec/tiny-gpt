#!/bin/bash
#python main.py --distributed=0 --lr3=0.0001
python main.py --distributed=0 --lr3=0.0003
python main.py --distributed=0 --lr3=0.001 --wd=0.01
python main.py --distributed=0 --lr3=0.001 --wd=0.1
python main.py --distributed=0 --lr3=0.003
python main.py --distributed=0 --lr3=0.01
python main.py --distributed=0 --lr3=0.03