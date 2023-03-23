# MBDF
Python script for generating the local Many Body Distribution Functionals (MBDF) and global Density of Functionals (DF) representations.
It also contains functions for generating the Coulomb Matrix (CM) and Bag of Bonds (BOB) representations.
MBDF is a local representation (atomic) and works with local kernels while DF is its global form and works with global kernels. It is recommended to not vectorize the MBDF representation if a global representation is required. DF should be used instead.

# Dependencies
Python libraries required : 
* Numpy
* Numba
* Joblib

If a progress bar is desired during the representation generation then the `tqdm` library is also required.

# Usage
To generate the **local** MBDF representation for your entire dataset:
```
from MBDF import generate_mbdf
rep = generate_mbdf(charges, coordinates)
```
Where `charges` is an array containing lists (or arrays) of nuclear charges for all molecules in the dataset. Likewise, `coordinates` should be an array containing lists (or arrays) of atomic coordinates for all molecules. 

The local cutoff distance can be controlled using the `cutoff_r` keyword. The default value is 8 Å but this should be increased for larger molecules. This does not affect the representation size, kernel evualuation cost and only affects the representation generation cost.

Note : The atomic coordinates should be provided in Angstrom.

The `n_jobs` parameter controls the number of cores over which the representation generation will be parallelized. Default value is `-1` which means all cores in the system will be used.

A progress bar for the representation generation process can be obtained by passing the parameter `progress_bar = True` to the function above. This requires the `tqdm` library.

It is recommended that the MBDF arrays be generated for the entire dataset (train & test) together since the functional values are normalized w.r.t their maximum in the dataset. This makes hyperparameter selection (length scales) easier when using Kernel based methods. The normalization can be turned off using `normalized = False`

To generate the **global** Density of functionals representation the MBDF array is required :
```
from MBDF import generate_df
rep = generate_df(mbdf, charges)
```
Where `mbdf` is the array containing the MBDF representation for all molecules in the dataset and `charges` is an array containing lists (or arrays) of nuclear charges for all molecules in the dataset.
The `binsize` keyword controls the grid-spacing used for discretizing the density function. The default value is 0.2 but this can be lowered if a higher resolution is required. It makes the representation perform better but increases its size.
