#!/bin/bash
echo "Begin to search !"
for gamma_guess in  -3.0  0.0 3.0 #要过sigmoid
do
    for corr in -100 -10 0 1 5 10 #要过softplus
    do
        echo "the gamma_guess is $gamma_guess"
        echo "the corr is $corr"
        python run_models_covid.py --heavyBall --gamma_guess $gamma_guess --corr $corr
    done
done
