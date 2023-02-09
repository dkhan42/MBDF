# MBDF
Python script for generating the local Many Body Distribution Functionals (MBDF) and global Density of Functionals (DF) representations.
It also contains functions for generating the Coulomb Matrix (CM) and Bag of Bonds (BOB) representations.

# Dependencies
Python libraries required : 
* Numpy
* Numba
* Joblib

If a progress bar is desired during the representation generation then the `tqdm` library is also required.

# Usage
To generate the local MBDF representation for your entire dataset:
```
import MBDF
rep = MBDF.generate_mbdf(charges, coordinates, n_jobs)
```
Where `charges` is an array containing lists (or arrays) of nuclear charges for all molecules in the dataset. Likewise, `coordinates` should be an array containing lists (or arrays) of atomic coordinates for all molecules. 
\
Note : The atomic coordinates should be in Angstrom. The representation hyperparameters were optimized in Bohr so the function automatically converts the coordinates from Angstrom to Bohr. The default local cutoff used is 12 Bohr which corresponds to about 6 Angstroms.

The `n_jobs` parameter controls the number of cores over which the representation generation will be parallelized. Default value is `-1` which means all cores in the system will be used.

A progress bar for the representation generation process can be obtained by passing the parameter `progress = True` to the function above. This requires the `tqdm` library.

It is recommended that the MBDF arrays be generated for the entire dataset (train & test) together since the functional values are normalized w.r.t their maximum in the dataset. This makes hyperparameter selection (length scales) easier when using Kernel based methods. The normalization can be turned off using `normalized = False`

To generate the global Density of functionals representation the MBDF array is required :
```
import MBDF
rep = MBDF.generate_DF(mbdf, charges, n_jobs)
```
Where `mbdf` is the array containing the MBDF representation for all molecules in the dataset and `charges` is an array containing lists (or arrays) of nuclear charges for all molecules in the dataset.
