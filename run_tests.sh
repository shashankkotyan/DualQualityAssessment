#!/usr/bin/env bash

for attack in pixel threshold
do
    for es in 1 0
    do
        for family in 1
        do
            for dataset in 0
            do
                for model in 3
                do
                    python -u code/run_attack.py $attack --gpu_index 0 -f $family -d $dataset -m $model --samples 10000 --epochs 200 -v --defence 0 --augmentation
                done
            done
        done
    done
done

