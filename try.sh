#!/bin/bash

# heavyBall & Base (low tol)
# for _ in 1 2 3 4 5
# do 
#     python run_models_covid.py --heavyBall --gamma_guess 0.0 --corr 5 --solver dopri5 --rtol 1e-2 --atol 1e-4
#     python run_models_covid.py --solver dopri5 --rtol 1e-2 --atol 1e-4
# done

# search cheb and atten Layer
for layer in 1 2
do
    for _ in 1 2 3
    do
        # python run_models_covid.py --rec-layers $layer --encoder Cheb
        python run_models_covid.py --rec-layers $layer --encoder Cheb_cat
        # python run_models_covid.py --rec-layers $layer
    done
done
