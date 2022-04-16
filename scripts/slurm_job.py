#!/usr/bin/python

import os

def makejob(model, nruns):
    return f"""#!/bin/bash

#SBATCH --job-name=trocr-{model}
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}
#SBATCH -w sh07

python3 train.py --level sentences --test_size 0.1 --num_beams 4 --limit_eval 256 --per_device_train_batch_size 32 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2
"""

def submit_job(job):
    with open('job.sbatch', 'w') as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure the log directory exists
os.system("mkdir -p logslurms")

# Launch the batch jobs
submit_job(makejob("sep", 1))