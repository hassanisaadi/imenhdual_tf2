#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=5G
#SBATCH --time=0-02:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.create_train.out
module load python/2.7.14
source /home/hassanih/project/hassanih/ENV/bin/activate
./create_train.py
