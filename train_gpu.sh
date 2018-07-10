#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --time=0-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train_gpu.out
#SBATCH --job-name=trgpu
echo "Starting run at: `date`"
log_dir="./logs-gpu/"
source /home/hassanih/project/hassanih/ENV/bin/activate
tensorboard --logdir=$log_dir --port=8009 &
./main.py\
  --epoch 200\
  --batch_size 128\
  --lr 0.001\
  --use_gpu 1\
  --phase train\
  --checkpoint_dir ./checkpoint_gpu\
  --sample_dir ./eval_results_gpu\
  --test_dir ./test_results_gpu\
  --eval_set ./data/eval\
  --eval_every_epoch 1\
  --PARALLAX 64\
  --hdf5_file ./data/data_da2_p33_s24_b128_par64_tr190.hdf5
