# Launch python code
start=`date +%s`
init_emc.py --qmin 20000000 --qmax 43644692 -d ../../data/data.cxi


for i in {1..2}; do
    logR.py --rc 1024 --dc 1024 -M 7 -d ../../data/data.cxi
    calculate_probabilities.py --beta 0.02
    update_I.py --ic 1024 -d ../../data/data.cxi
done


end=`date +%s`
runtime=$((end-start))
echo 
echo runtime: "$runtime" seconds
