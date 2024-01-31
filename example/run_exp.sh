#!/bin/bash
for t in 8192
do
    for i in 500
    do
        python example.py --ntips $t --epochs $i --niter 100000
    done
done