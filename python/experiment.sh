#!/bin/bash
#
#SBATCH --job-name=experiment.sh
#SBATCH --output=experiment.txt
#
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --mem-per-cpu=10000
#
#SBATCH --array=1-300

srun python experiment.py $SLURM_ARRAY_TASK_ID

# Copyright (c) 2018  Joris M. Mooij
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
