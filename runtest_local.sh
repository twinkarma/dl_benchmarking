#!/bin/bash
# set the number of nodes
#SBATCH -p small

# set max wallclock time
#SBATCH --time=10:00:00

# set name of job
#SBATCH --job-name=dl_model_comparison

# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source activate jupyter



python main.py "testdldata" "testoutput.csv" "disk"
python main.py "testdldata" "testoutput.csv" "flash"
