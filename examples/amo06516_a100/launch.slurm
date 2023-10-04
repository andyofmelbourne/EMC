#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=0-4:00:00
#SBATCH --mail-user=morganaj@unimelb.edu.au
#SBATCH --mail-type=ALL

# Load required modules
source /home/andyofmelbourne/.bashrc

# Launch python code
init_emc.py --qmin 20000000 --qmax -2 -m 125

for i in {1..20}; do
    logR.py --rc 1024 --dc 1024 -M 25
    calculate_probabilities.py --beta 0.02
    update_I.py --ic 1024 
done


#my-job-stats -a -n -s
