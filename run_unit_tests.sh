#!/usr/bin/env bash

# Check Models
for attack in fglinf
do
    for family in 0
    do
        for dataset in 0 
        do
            for model in 0 1
            do
                python -u code/run_attack.py $attack -f $family -d $dataset -m $model --samples 1 --epochs 2 -v 
            done
        done
    done

    for family in 1
    do
        for dataset in 0
        do
            for model in 1 2 3 4 5 6 7
            do
                python -u code/run_attack.py $attack -f $family -d $dataset -m $model --samples 1 --epochs 2 -v
            done
        done
    done
done

#Check Our Attacks
for attack in pixel threshold 
do
    for es in 1 0
    do
        for family in 0
        do
            for dataset in 0
            do
                for model in 0
                do
                    python -u code/run_attack.py $attack -e $es -f $family -d $dataset -m $model --samples 1 --epochs 1 -th 64 -v 
                done
            done
        done

        for family in 1
        do
            for dataset in 0
            do
                for model in 1
                do
                    python -u code/run_attack.py $attack -e $es -f $family -d $dataset -m $model --samples 1 --epochs 1 -th 64 -v
                done
            done
        done
    done
done

#Check Other Attacks
for attack in bilinf pgdlinf deep newton
do
        for family in 0
        do
            for dataset in 0
            do
                for model in 0
                do
                    python -u code/run_attack.py $attack -f $family -d $dataset -m $model --samples 1 --epochs 2 -v 
                done
            done
        done

        for family in 1
        do
            for dataset in 0
            do
                for model in 1
                do
                    python -u code/run_attack.py $attack -f $family -d $dataset -m $model --samples 1 --epochs 2 -v
                done
            done
        done
done
