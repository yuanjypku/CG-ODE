#!/bin/bash
echo "Begin to search !"
for graf_layer in 1 2 3 4 5
do
    echo "the graf_layer is $graf_layer"
    python run_models_covid.py --graf_layer $graf_layer
done
# python run_models_covid.py --graf_layer 1