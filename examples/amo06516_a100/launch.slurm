#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=0-4:00:00
#SBATCH --mail-user=morganaj@unimelb.edu.au
#SBATCH --mail-type=ALL

set -e

# Load required modules
source /home/andyofmelbourne/.bashrc

# Launch python code
#init_emc.py --qmin 20000000 --qmax -2 -m 125

for i in {1..5}; do
    logR.py -M 32 --rc 50000 --ic 4096 
    calculate_probabilities.py --beta 0.02 
    mpirun -np 4 update_I.py --mpx 129 --qmin 0 --qmax -2
done

# continuing from last time
#mpirun -np 4 update_I.py --ic 512 
    
#for i in {1..3}; do
#    logR.py --ic 1024 -M 25
#    calculate_probabilities.py --beta 0.02
#    mpirun -np 4 update_I.py --ic 512 
#done

# just merge in I once, then compare likelihoods
# .211 for normal merge and .213 for I merge (better)
#mpirun -np 4 update_I.py --ic 512 --qmin 20000000 --qmax -2 -m 125 -o merged_intensity.pickle.no_cosmic_I_merge

# look at LL
#logR.py --ic 1024 -M 25 --merged_intensity merged_intensity_I_merge.pickle
#calculate_probabilities.py --beta 0.02 -P probability-matrix-merged_intensity_I_merge.h5

#logR.py --ic 1024 -M 25 --merged_intensity merged_intensity_normal_merge.pickle
#calculate_probabilities.py --beta 0.02 -P probability-matrix-merged_intensity_normal_merge.h5


##Log this job's resource usage stats###
my-job-stats -a -n -s
##
