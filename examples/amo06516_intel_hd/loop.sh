#!/bin/bash
# Launch python code
start=`date +%s`
#init_emc.py --qmin 20000000 --qmax 43644692 


for i in {1..4}; do
    logR.py --ic 1024 -M 7 
    calculate_probabilities.py --beta 0.02
    update_I.py --ic 1024 
done


end=`date +%s`
runtime=$((end-start))
echo 
echo runtime: "$runtime" seconds
