#!/bin/bash
for _ in in 1 2 3 4 5
do 
    python run_models_covid.py --heavyBall --gamma_guess 0.0 --corr 5 --solver dopri5 --rtol 1e-2 --atol 1e-4
    python run_models_covid.py --solver dopri5 --rtol 1e-2 --atol 1e-4
done
