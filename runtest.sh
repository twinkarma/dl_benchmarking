#!/bin/bash
# set the number of nodes
#SBATCH -p small

# set max wallclock time
#SBATCH --time=48:00:00

# set name of job
#SBATCH --job-name=dl_model_comparison

# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --mem=32G


module load python/anaconda3
source activate testtf

cd ~/dl_benchmarking

python main.py "/jmain02/home/HCBEACH01/dxl08/txk31-dxl08/testdldata" "/jmain02/home/HCBEACH01/dxl08/txk31-dxl08/dldatasetoutput.csv" "disk"
python main.py "/jmain02/flash/HCBEACH01/dxl08/txk31-dxl08/testdldata" "/jmain02/home/HCBEACH01/dxl08/txk31-dxl08/dldatasetoutput.csv" "flash"


