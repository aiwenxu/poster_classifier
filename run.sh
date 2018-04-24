#!/bin/bash -l

#SBATCH --job-name=poster_classifier
#SBATCH --nodes=1
#SBATCH --time=24:00:00 # Runtime in D-HH:MMs
#SBATCH --partition=aquila # Partition to submit to
#SBATCH --constraint=2680v4
#SBATCH --mem=20000 # Memory pool for all cores, MB
#SBATCH --output=output.slurm
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ax266@nyu.edu # Email or Chinese mobile phone NO. to which notifications will be sent
#SBATCH --gres=gpu:k80:1
#SBATCH --share

cd /gpfsnyu/scratch/ax266/poster_classifier
python train_classifier.py