#!/bin/bash
#
#SBATCH --job-name=explonb
#
#SBATCH --partition=keira
#SBATCH --nodes=1
#SBATCH --exclusive
##SBATCH	--exclude=mb-rom[203-206]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
# #SBATCH --mem-per-cpu=850
#SBATCH --mem=0
#SBATCH --output=./job.out
#   #SBATCH --error=./job.err
#

export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/venvs/modnenv_v2/bin/activate

# ulimit -s unlimited

python test.py >> ./job.out
echo "DONE"
