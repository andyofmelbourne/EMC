#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=0-0:40:00
#SBATCH --mail-user=morganaj@unimelb.edu.au
#SBATCH --mail-type=ALL

# Load required modules
#source /home/andyofmelbourne/.bashrc.bak

# Launch python code
python init_emc.py --qmin 20000000 --qmax 43644692

for i in {1..10}; do
    python logR.py --rc 1024 --dc 1024 -M 10
    python calculate_probabilities.py --beta 0.02
    python update_I_split_pixels.py --ic 1024 
done


#my-job-stats -a -n -s
