# sigmasep
Code for UAI 2018 paper by Forré &amp; Mooij (causal discovery with mSCMs using sigma-separation)

## Version
v1.1 

## ChangeLog
- v1.1: bug fix (added one missing sigma-sep rule in sigma_hej_cyclic.pl)

## License
This code is licensed under the BSD 2-clause license (see file LICENSE).

## Citation
When making significant use of this code for a scientific publication, please cite:

    @inproceedings{ForreMooij_UAI_18,
      author    = {Patrick Forr{\'e} and Joris M. Mooij},
      title     = {Constraint-based Causal Discovery for Non-Linear Structural Causal Models with Cycles and Latent Confounders},
      booktitle = {Proceedings of the 34th Annual Conference on {U}ncertainty in {A}rtificial {I}ntelligence ({UAI}-18)},
      year      = 2018
    }

A significant part of the code is based on the code accompanying the following paper:

    @inproceedings{Hyttinen++2014,
      author    = {Hyttinen, A. and Eberhardt, F. and J{\"{a}}rvisalo, M.},
      title     = {Constraint-based Causal Discovery: Conflict Resolution with Answer Set Programming},
      booktitle = {Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence, ({UAI}-14)},
      address   = {Quebec City, Quebec, Canada},
      pages     = {340--349},
      year      = {2014}
    }

## Getting started
The code isn't designed to run with one keystroke or be user-friendly. It should however
be helpful to reproduce the experiments reported in our paper, and as a starting point
for a more user-friendly implementation.

To reproduce the experiments reported in the paper, look into the python notebook 
`python/Experiments_and_Plotting.ipynb`

## Frequently Asked Questions
None so far. For questions, you can email the authors.
