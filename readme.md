
# ABOUT THE PACKAGE


FRISBHEE - FRIedmann Solver for Black Hole Evaporation in the Early-universe
Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner

Python package providing the solution of the Friedmann - Boltzmann equations for Primordial Black 
Holes + SM radiation + BSM Models + inflaton field.


# FRISBHEE + Inflaton decay

Modificatins to include inflaton in the evolution
The main part is FRISBHEE has been modified to include the decay/scattering of inflaton in the dynamical
evolving system. It allows to study all scenarios of reheating. Inflaton decays into SM only.
Author: D. Kpatcha 


This package provides the solution of the Friedmann - Boltzmann equations for Primordial Black 
Holes + Inflaton + SM radiation + BSM Models. We consider the collapse of density fluctuations as the PBH 
formation mechanism. We provide codes for monochromatic and extended mass and spin distributions.


# CONTAINS

	Friedmann.py: The main classes, return the full evolution of the PBH, SM and Dark Radiation 
                      comoving energy densities, together with the evolution of the PBH mass and spin as function of
                      the scale factor. 

	Functions_Kerr_Power_Law.py: Contains essential functions for the power low PBH mass distribution

	Functions_phik.py: Contains essential functions related to the evolution of the inflaton

	BHProp.py: contains essential constants and parameters

	script_scan.py: main programm of the package. Read parameters from the command line input
                       can be run in terminal (see "run.sh" or "run_script.sh")

	run.sh: bash shell script to run the programm for specific PBH initial mass and infaton potential.
                input paramters (e.g initial fraction of BH energy density beta, infaton decay coupling, ...) 
                are defined.

	run_script.sh: bash shell script to run the programm for various (scan) PBH initial mass and infaton potential.
               	input paramters (e.g initial fraction of BH energy density beta, infaton decay coupling, ...) 
                       are defined.


# INSTALLATION AND RUNNING

	1. No specific installation instructions needed. just required modules (see below)
	
	2. For running see "./run.sh" or "./run_script.sh"
	   The results (output files) are stored in 
	   ./Results/k=$kn/phiff/$yuk/$MBHin/databeta=$beta/sigma_$sigmaM/


# REQUIRED MODULES

We use Pathos (https://pathos.readthedocs.io/en/latest/pathos.html) for parallelisation, and 
tqdm (https://pypi.org/project/tqdm/) for progress meter. These should be installed in order 
to ReheatingPyTools to run.


# CREDITS

If using this code, please cite:
    arXiv:2107.00013, arXiv:2107.00016, arXiv:2207.09462, arXiv:2212.03878, arXiv:2305.10518, arXiv:2309.06505

