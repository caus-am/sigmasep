# Copyright (c) 2018  Patrick Forre, Joris M. Mooij
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import mSCM
import sys
import numpy as np
from numpy.random import choice
from numpy.random import seed
import random

nbr = int(sys.argv[1])
random.seed(nbr)
np.random.seed(nbr)

# change the following to point to the directory in which you have cloned the code repository
rootdir = "~/vcs/sigmasep"
# change the following to point to the directory in which you want to save the output
outdir = "/dev/shm/jmooij1/sigmasep"
# change the following to point to the directory in which clingo lives
clingodir = "/zfs/ivi/causality/opt/clingo-4.5.4-linux-x86_64/"

for nbr_do in range(6):
    mSCM.sample_mSCM_run_all_and_save(
        d=5,k=2,p=0.3,m=0,nbr=nbr,add_ind_noise_to_A=False,
        add_ind_noise_to_W=True,
        include_latent=True,
        folderpath=outdir+"/mSCM_data/experiment_"+str(nbr_do)+"/",
        AF=[np.tanh],SC=[1],NOI=['normal'],SD=[1],n=10000,
        AL =[0.001],MUL=[1000],infty=1000,nbr_do=nbr_do,max_do=1,do_strategy=2,
        clingodir=clingodir,
        aspdir=rootdir+"/ASP/"
    )
