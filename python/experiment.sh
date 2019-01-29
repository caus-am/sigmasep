#!/bin/bash
#
#SBATCH --job-name=sigmasep
#SBATCH --output=/dev/shm/jmooij1/sigmasep/log/sigmasep-%a.stdout
#SBATCH --error=/dev/shm/jmooij1/sigmasep/log/sigmasep-%a.stderr
#
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --mem-per-cpu=1000
#
#SBATCH --array=1-300

python experiment.py $SLURM_ARRAY_TASK_ID

# Copyright (c) 2018  Joris M. Mooij
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
