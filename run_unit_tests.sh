#!/usr/bin/env bash

for attack in pixel threshold 
do
    for es in 0 1
    do
        for family in 0
        do
            for dataset in 0 1
            do
                for model in 0 1
                do
                    python -u run_attack.py $attack -e $es -f $family -d $dataset -m $model -s 1 --epochs 2 -v
                done
            done
        done

        for family in 1
        do
            for dataset in 0 1
            do
                for model in 0 1 2 3 4 5 6 7
                do
                    python -u run_attack.py $attack -e $es -f $family -d $dataset -m $model -s 1 --epochs 2 -v
                done
            done
        done

    done
done
