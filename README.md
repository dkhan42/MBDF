# MBDF
Python script for generating the Many Body Distribution Functionals (local) and Density of Functionals (global) representations.
It also contains functions for generating the Coulomb Matrix and Bag of Bonds representations.

# Dependencies
Python libraries required : 
* Numpy
* Numba
* Joblib

If a progress bar is desired during the representation generation then the `tqdm` library is also required.

# Usage
It is highly recommended that the MBDF arrays be generated for the entire dataset (train & test) at once since the functional values are normalized w.r.t the largest and smallest values in the dataset. This makes it easier to 
