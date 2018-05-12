#!/bin/bash -l

#SBATCH --job-name=poster_classifier
#SBATCH --nodes=1
#SBATCH --time=6:00:00 # Runtime in D-HH:MMs
#SBATCH --partition=gpu # Partition to submit to
#SBATCH --mem=20000 # Memory pool for all cores, MB
#SBATCH --output=test_output.slurm
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ax266@nyu.edu # Email or Chinese mobile phone NO. to which notifications will be sent
#SBATCH --gres=gpu:1  # To request GPU, include this line. You can specify kind and number of GPU.
#SBATCH --share

python test.py