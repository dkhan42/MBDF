import numpy as np
import numba
from copy import deepcopy

root2,ipi=2**0.5,np.pi*1j
half_rootpi=(np.pi**0.5)/2
c1,c2,c3=4.08858*(10**12),(np.pi**0.5)/2,(np.pi**0.5)*np.exp(-0.25)*1j/4
c4=-1j*(np.pi**0.5)*np.exp(-1/8)/(4*root2)
a2b = 1.88973


@numba.jit(nopython=True)
def erfunc(z):
    t = 1.0 / (1.0 + 0.5 * np.abs(z))
    ans = 1 - t * np.exp( -z*z -  1.26551223 +
                        t * ( 1.00002368 +
                        t * ( 0.37409196 + 
                        t * ( 0.09678418 + 
                        t * (-0.18628806 + 
                        t * ( 0.27886807 + 
                        t * (-1.13520398 + 
                        t * ( 1.48851587 + 
                        t * (-0.82215223 + 
                        t * ( 0.17087277))))))))))
    return ans


@numba.jit(nopython=True)
def hermite_polynomial(x, degree, a=1):
    if degree == 0:
        return 1
    elif degree == 1:
        return -2*a*x
    elif degree == 2:
        x1 = (a*x)**2
        return 4*x1 - 2*a
    elif degree == 3:
        x1 = (a*x)**3
        return -8*x1 - 12*a*x
    elif degree == 4:
        x1 = (a*x)**4
        x2 = (a*x)**2
        return 16*x1 - 48*x2 + 12*a**2


@numba.jit(nopython=True)
def generate_data(size,z,atom,charges,coods,cutoff_r=12):
    """
    returns 2 and 3-body internal coordinates
    """
    
    twob=np.zeros((size,2))
    threeb=np.zeros((size,size,5))
    z1=z**0.8

    for j in range(size):
        rij=atom-coods[j]
        rij_norm=np.linalg.norm(rij)

        if rij_norm!=0 and rij_norm<cutoff_r:
            z2=charges[j]**0.8
            twob[j]=rij_norm,z1*charges[j]

            for k in range(size):
                if j!=k:
                    rik=atom-coods[k]
                    rik_norm=np.linalg.norm(rik)

                    if rik_norm!=0 and rik_norm<cutoff_r:
                        z3=charges[k]**0.8
                        
                        rkj=coods[k]-coods[j]
                        
                        rkj_norm=np.linalg.norm(rkj)
                        
                        threeb[j][k][0] = np.minimum(1.0,np.maximum(np.dot(rij,rik)/(rij_norm*rik_norm),-1.0))
                        threeb[j][k][1] = np.minimum(1.0,np.maximum(np.dot(rij,rkj)/(rij_norm*rkj_norm),-1.0))
                        threeb[j][k][2] = np.minimum(1.0,np.maximum(np.dot(-rkj,rik)/(rkj_norm*rik_norm),-1.0))
                        
                        atm = rij_norm*rik_norm*rkj_norm
                        
                        charge = z1*z2*z3
                        
                        threeb[j][k][3:] =  atm, charge

    return twob, threeb                        

@numba.jit(nopython=True)
def angular_integrals(size,threeb,alength=158,a=2,grid1=None,grid2=None,angular_scaling=2.4):
    """
    evaluates the 3-body functionals using the trapezoidal rule
    """

    arr=np.zeros((alength,2))
    theta=0
    
    for i in range(alength):
        f1,f2=0,0
        num1,num2=grid1[i],grid2[i]
        
        for j in range(size):

            for k in range(size):

                if threeb[j][k][-1]!=0:
                    
                    angle1,angle2,angle3,atm,charge=threeb[j][k]

                    x=theta-np.arccos(angle1)

                    exponent,h1=np.exp(-a*x**2),hermite_polynomial(x,1,a)
                    
                    f1+=(charge*exponent*num1)/(atm**4)
                    
                    f2+=(charge*h1*exponent*(1+(num2*angle1*angle2*angle3)))/(atm**angular_scaling)
        
        arr[i]=f1,f2
        theta+=0.02

    trapz=[np.trapz(arr[:,i],dx=0.02) for i in range(arr.shape[1])]

    return trapz


@numba.jit(nopython=True)
def radial_integrals(size,rlength,twob,step_r,a=1,normalized=False):
    """
    evaluates the 2-body functionals using the trapezoidal rule
    """
    
    arr=np.zeros((rlength,4))
    r=0
    
    for i in range(rlength):
        f1,f2,f3,f4=0,0,0,0
        
        for j in range(size):

            if twob[j][-1]!=0:
                dist,charge=twob[j]
                x=r-dist

                if normalized==True:
                    norm=(erfunc(dist)+1)*half_rootpi
                    exponent=np.exp(-a*(x)**2)/norm
                
                else:
                    exponent=np.exp(-a*(x)**2)

                h1,h2=hermite_polynomial(x,1,a),hermite_polynomial(x,2,a)
                
                f1+=charge*exponent*np.exp(-10.8*r)
                
                f2+=charge*exponent/(2.2508*(r+1)**3)
                
                f3+=charge*(h1*exponent)/(2.2508*(r+1)**6)
                
                f4+=charge*h2*exponent*np.exp(-1.5*r) 
        
        r+=step_r
        arr[i]=f1,f2,f3,f4
    
    trapz=[np.trapz(arr[:,i],dx=step_r) for i in range(arr.shape[1])]

    return trapz


@numba.jit(nopython=True)
def mbdf_local(charges,coods,grid1,grid2,rlength,alength,pad=29,step_r=0.1,cutoff_r=12,angular_scaling=2.4):
    """
    returns the local MBDF representation for a molecule
    """
    size = len(charges)
    mat=np.zeros((pad,6))
    
    assert size > 1, "No implementation for monoatomics"

    if size>2:
        for i in range(size):

            twob,threeb = generate_data(size,charges[i],coods[i],charges,coods,cutoff_r)

            mat[i][:4] = radial_integrals(size,rlength,twob,step_r)     

            mat[i][4:] = angular_integrals(size,threeb,alength,grid1=grid1,grid2=grid2,angular_scaling=angular_scaling)

    elif size==2:
        z1, z2, rij = charges[0]**0.8, charges[1]**0.8, coods[0]-coods[1]
        
        pref, dist = z1*z2, np.linalg.norm(rij)
        
        twob = np.array([[pref, dist], [pref, dist]])
        
        mat[0][:4] = radial_integrals(size,rlength,twob,step_r)

        mat[1][:4] = mat[0][:4]

    return mat


def mbdf_global(charges,coods,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r=0.1,cutoff_r=12,angular_scaling=2.4):
    """
    returns the flattened, bagged MBDF feature vector for a molecule
    """
    elements = {k:[[],k] for k in keys}

    size = len(charges)

    for i in range(size):
        elements[charges[i]][0].append(coods[i])

    mat, ind = np.zeros((rep_size,6)), 0

    assert size > 1, "No implementation for monoatomics"

    if size>2:

        for key in keys:
            
            num = len(elements[key][0])
            
            if num!=0:
                bags = np.zeros((num,6))
                
                for j in range(num):
                    twob,threeb = generate_data(size,key,elements[key][0][j],charges,coods,cutoff_r)

                    bags[j][:4] = radial_integrals(size,rlength,twob,step_r)     

                    bags[j][4:] = angular_integrals(size,threeb,alength,grid1=grid1,grid2=grid2,angular_scaling=angular_scaling)

                mat[ind:ind+num] = -np.sort(-bags,axis=0)
                
            ind += asize[key]
    
    elif size == 2:

        for key in keys:
            
            num = len(elements[key][0])
            
            if num!=0:
                bags = np.zeros((num,6))
                
                for j in range(num):
                    z1, z2, rij = charges[0]**0.8, charges[1]**0.8, coods[0]-coods[1]
        
                    pref, dist = z1*z2, np.linalg.norm(rij)

                    twob = np.array([[pref, dist], [pref, dist]])
                    
                    bags[j][:4] = radial_integrals(size,rlength,twob,step_r)     

                mat[ind:ind+num] = -np.sort(-bags,axis=0)
                
            ind += asize[key]

    return mat
                        

@numba.jit(nopython=True)
def fourier_grid():
    
    angles = np.arange(0,np.pi,0.02)
    
    grid1 = np.cos(angles)
    grid2 = np.cos(2*angles)
    grid3 = np.cos(3*angles)
    
    return (3+(100*grid1)+(-200*grid2)+(-164*grid3),grid1)


@numba.jit(nopython=True)
def normalize(A,normal='mean'):
    """
    normalizes the functionals based on the given method
    """
    
    A_temp = np.zeros(A.shape)
    
    if normal=='mean':
        for i in range(A.shape[2]):
            
            avg = np.mean(A[:,:,i])

            if avg!=0.0:
                A_temp[:,:,i] = A[:,:,i]/avg
            
            else:
                pass
   
    elif normal=='min-max':
        for i in range(A.shape[2]):
            
            diff = np.abs(np.max(A[:,:,i])-np.min(A[:,:,i]))
            
            if diff!=0.0:
                A_temp[:,:,i] = A[:,:,i]/diff
            
            else:
                pass
    
    return A_temp


from joblib import Parallel, delayed

def generate_mbdf(nuclear_charges,coords,local=True,n_jobs=-1,pad=None,step_r=0.1,cutoff_r=8.0,step_a=0.02,angular_scaling=4,normalized='min-max',progress_bar=False):
    """
    Generates the local MBDF representation arrays for a set of given molecules

    :param nuclear_charges: array of arrays of nuclear_charges for all molecules in the dataset
    :type nuclear_charges: numpy array NxM, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param coords : array of arrays of input coordinates of the atoms
    :type coords: numpy array NxMx3, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    ordering of the molecules in the nuclear_charges and coords arrays should be consistent
    :param n_jobs: number of cores to parallelise the representation generation over. Default value is -1 which uses all available cores in the system
    :type n_jobs: integer
    :param pad: Number of atoms in the largest molecule in the dataset. Can be left to None and the function will calculate it using the nuclear_charges array
    :type pad: integer
    :param step_r: radial step length in Angstrom
    :type step_r: float
    :param cutoff_r: local radial cutoff distance for each atom
    :type cutoff_r: float
    :param step_a: angular step length in Radians
    :type step_a: float
    :param angular_scaling: scaling of the inverse distance weighting used in the angular functionals
    :type : float
    :param normalized: type of normalization to be applied to the functionals. Available options are 'min-max' and 'mean'. Can be turned off by passing False
    :type : string
    :param progress: displays a progress bar for representation generation process. Requires the tqdm library
    :type progress: Bool

    :return: NxPadx6 array containing Padx6 dimensional MBDF matrices for the N molecules
    """
    assert nuclear_charges.shape[0] == coords.shape[0], "charges and coordinates array length mis-match"
    
    lengths, charges = [], []

    for i in range(len(nuclear_charges)):
        
        q, r = nuclear_charges[i], coords[i]
        
        assert q.shape[0] == r.shape[0], "charges and coordinates array length mis-match for molecule at index" + str(i)

        lengths.append(len(q))

        charges.append(q.astype(np.float64))

    if pad==None:
        pad = max(lengths)

    #charges = np.array(charges)

    rlength = int(cutoff_r/step_r) + 1
    alength = int(np.pi/step_a) + 1

    grid1,grid2 = fourier_grid()
    
    coords, cutoff_r = a2b*coords, a2b*cutoff_r

    if local:
        if progress_bar==True:

            from tqdm import tqdm    
            mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_local)(charge,cood,grid1,grid2,rlength,alength,pad,step_r,cutoff_r,angular_scaling) for charge,cood in tqdm(list(zip(charges,coords))))

        else:
            mbdf = Parallel(n_jobs=n_jobs)(delayed(mbdf_local)(charge,cood,grid1,grid2,rlength,alength,pad,step_r,cutoff_r,angular_scaling) for charge,cood in zip(charges,coords))

        mbdf=np.array(mbdf)

        if normalized==False:

            return mbdf

        else:

            return normalize(mbdf,normal=normalized)
        
    else:
        keys = np.unique(np.concatenate(charges))

        asize = {key:max([(mol == key).sum() for mol in charges]) for key in keys}

        rep_size = sum(asize.values())

        if progress_bar==True:

            from tqdm import tqdm    
            arr = Parallel(n_jobs=n_jobs)(delayed(mbdf_global)(charge,cood,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r,cutoff_r,angular_scaling) for charge,cood in tqdm(list(zip(charges,coords))))

        else:
            arr = Parallel(n_jobs=n_jobs)(delayed(mbdf_global)(charge,cood,asize,rep_size,keys,grid1,grid2,rlength,alength,step_r,cutoff_r,angular_scaling) for charge,cood in zip(charges,coords))

        arr = np.array(arr)

        if normalized==False:

            mbdf = np.array([mat.ravel(order='F') for mat in arr])
            
            return mbdf

        else:

            arr = normalize(arr,normal=normalized)

            mbdf = np.array([mat.ravel(order='F') for mat in arr])
            
            return mbdf


@numba.jit(nopython=True)
def wKDE(rep,bin,bandwidth,kernel,scaling=False):
    """
    returns the weighted kernel density estimate for a given array and bins
    """
    if kernel=='gaussian':
        if scaling=='root':
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(a**2)/bandwidth)
            
            k = (np.sqrt(np.abs(rep)))*basis
            
            return np.sum(k,axis=1)

        else:
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(a**2)/bandwidth)
            
            return np.sum(basis,axis=1)

    elif kernel=='laplacian':
        if scaling=='root':
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(np.abs(a))/bandwidth)
            
            k = (np.abs(rep))*basis
            
            return np.sum(k,axis=1)

        else:
            a = bin.reshape(-1,1)-rep
            
            basis = np.exp(-(np.abs(a))/bandwidth)
            
            return np.sum(basis,axis=1)


def density_estimate(reps,nuclear_charges,keys,bin,bandwidth,kernel='gaussian',scaling='root'):
    """
    returns the density functions of MBDF functionals for a set of given molecules.
    """
    
    size=len(bin)
    big_rep=np.zeros((reps.shape[0],size*len(keys)))
    

    if kernel=='gaussian':
        for i in range(len(nuclear_charges)):

            for j,k in enumerate(keys):
                ii = np.where(nuclear_charges[i] == k)[0]

                if len(ii)!=0:
                    big_rep[i,j*size:(j+1)*size]=wKDE(reps[i][ii]/k,bin,bandwidth,kernel,scaling)

                else:
                    big_rep[i,j*size:(j+1)*size]=np.zeros(size)

    return big_rep


def generate_df(mbdf,nuclear_charges,bw=0.07,binsize=0.2,kernel='gaussian'):
    """
    Generates the Density of Functionals representation for a given set of molecules. Requires their MBDF arrays as input
    
    :param mbdf: array of arrays containing the MBDF representation matrices for all molecules in the dataset
    :type mbdf: numpy array, output of the generate_mbdf function can be directly used here
    :param nuclear_charges: array of arrays of nuclear_charges for all molecules in the dataset, should be in the same order as in the MBDF arrays
    :type nuclear_charges: numpy array NxM, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param bw: the bandwidth hyper-parameter of the kernel density estimate
    :type bw: float
    :param binsize: grid-spacing used for discretizing the density function
    :type binsize: float
    :param kernel: kernel function to be used in the kernel density estimation
    :type kernel: string

    :return: NxM array containing the M dimensional representation vectors for N molecules
    """
    fs=mbdf.shape[-1]

    reps=[10*mbdf[:,:,i]/(np.max(np.abs(mbdf[:,:,i]))) for i in range(fs)]
    
    keys=np.unique(np.concatenate(nuclear_charges))
    
    bin=np.arange(-10,10,binsize)
    
    gridsize=len(keys)*len(bin)
    
    kde=np.zeros((mbdf.shape[0],gridsize*fs))
    
    for i in range(fs):
        kde[:,i*gridsize:(i+1)*gridsize]=density_estimate(reps[i],nuclear_charges,keys,bin,bw,kernel)
    
    return kde

@numba.jit(nopython=True)
def generate_CM(cood,charges,pad):
    size=len(charges)
    cm=np.zeros((pad,pad))
    for i in range(size):
        for j in range(size):
            if i==j:
                cm[i,j]=0.5*(charges[i]**(2.4))
            else:
                dist=np.linalg.norm(cood[i,:]-cood[j,:])
                
                cm[i,j]=(charges[i]*charges[j])/dist
    summation = np.array([sum(x**2) for x in cm])
    sorted_mat = cm[np.argsort(summation)[::-1,],:]    
    return sorted_mat.ravel()


from math import comb,cos
from itertools import combinations, product


def generate_bob(elements,coords,n_jobs=-1,asize={'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1}):
    """
    generates the Bag of Bonds representation
    :param elements: array of arrays of chemical element symbols for all molecules in the dataset
    :type elements: numpy array NxM, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param coords: array of arrays of input coordinates of the atoms
    :type coords: numpy array NxMx3, where N is the number of molecules and M is the number of atoms (can be different for each molecule)
    :param n_jobs: number of cores to parallelise the representation generation over. Default value is -1 which uses all cores in the system
    :type n_jobs: integer
    :param asize: The maximum number of atoms of each element type supported by the representation
    :type asize: dictionary

    :return: NxD array of D-dimensional BoB vectors for the N molecules
    :rtype: numpy array
    """
    from tqdm import tqdm

    bob_arr = Parallel(n_jobs=n_jobs)(delayed(bob)(atoms,coods,asize) for atoms,coods in tqdm(list(zip(elements,coords))))

    return np.array(bob_arr)

def bob(atoms,coods, asize={'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1}):
    keys=list(asize.keys()) 
    elements={'C':[[],6],'H':[[],1],'N':[[],7],'O':[[],8],'F':[[],9],'P':[[],15],'S':[[],16],'Cl':[[],17],'Br':[[],35],'I':[[],53]}
    for i in range(len(atoms)):
        elements[atoms[i]][0].append(coods[i])
    bob=[]
    for key in keys:
        num=len(elements[key][0])
        if num!=0:
            bag=np.zeros((asize[key]))
            bag[:num]=0.5*(elements[key][1]**2.4)
            bag=-np.sort(-bag)
            bob.extend(bag)
            for j in range(i,len(keys)):
                if i==j:
                    z=elements[key][1]
                    bag=np.zeros((comb(asize[key],2)))
                    vec=[]
                    for (r1,r2) in combinations(elements[key][0],2):
                        vec.append(z**2/np.linalg.norm(r1-r2))
                    bag[:len(vec)]=vec
                    bag=-np.sort(-bag)
                    bob.extend(bag)
                elif (i!=j) and (len(elements[keys[j]][0])!=0):
                    z1,z2=elements[key][1],elements[keys[j]][1]
                    bag=np.zeros((asize[key]*asize[keys[j]]))
                    vec=[]
                    for (r1,r2) in product(elements[key][0],elements[keys[j]][0]):
                        vec.append(z1*z2/np.linalg.norm(r1-r2))
                    bag[:len(vec)]=vec
                    bag=-np.sort(-bag)
                    bob.extend(bag)
                else:
                    bob.extend(np.zeros((asize[key]*asize[keys[j]])))
        else:
            bob.extend(np.zeros((asize[key])))
            for j in range(i,len(keys)):
                if i==j:
                    bob.extend(np.zeros((comb(asize[key],2))))
                else:
                    bob.extend(np.zeros((asize[key]*asize[keys[j]])))
    return np.array(bob) 

from scipy.spatial.distance import cityblock, euclidean
from scipy.stats import wasserstein_distance

def get_delta_local_kernel(A,B,Q1,Q2,sigma,kernel='laplacian'):
    
    n1, n2 = A.shape[0], B.shape[0]

    assert n1 == Q1.shape[0], "charges and representation array length mis-match"
    assert n2 == Q2.shape[0], "charges and representation array length mis-match"

    K = 0
    
    if kernel == 'laplacian':

        for i in range(n1):
            k=0
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = cityblock(A[i],B[j])
                    k += np.exp(-dist/sigma)
            K += k

    elif kernel == 'gaussian':

        for i in range(n1):
            k=0
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = euclidean(A[i],B[j])
                    k += np.exp(-dist/sigma)
            K += k

    elif kernel == 'wasserstein':

        for i in range(n1):
            k=0
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = wasserstein_distance(A[i],B[j])
                    k += np.exp(-dist/sigma)
            K += k
    
    return K

def get_min_local_kernel(A,B,Q1,Q2,sigma,kernel='laplacian'):
    
    n1, n2 = A.shape[0], B.shape[0]

    assert n1 == Q1.shape[0], "charges and representation array length mis-match"
    assert n2 == Q2.shape[0], "charges and representation array length mis-match"

    K1, K2 = 0, 0
    
    if kernel == 'laplacian':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = cityblock(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = cityblock(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)
        return min([K1,K2])

    elif kernel == 'gaussian':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = euclidean(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = euclidean(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)
        return min([K1,K2])

    elif kernel == 'wasserstein':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = wasserstein_distance(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = wasserstein_distance(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)

        return min([K1,K2])

def get_max_local_kernel(A,B,Q1,Q2,sigma,kernel='laplacian'):
    
    n1, n2 = A.shape[0], B.shape[0]

    assert n1 == Q1.shape[0], "charges and representation array length mis-match"
    assert n2 == Q2.shape[0], "charges and representation array length mis-match"

    K1, K2 = 0, 0
    
    if kernel == 'laplacian':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = cityblock(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = cityblock(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)
        return max([K1,K2])

    elif kernel == 'gaussian':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = euclidean(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = euclidean(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)
        return max([K1,K2])

    elif kernel == 'wasserstein':

        for i in range(n1):
            k= []
            for j in range(n2):
                q1, q2 = Q1[i], Q2[j]

                if q1==q2:
                    dist = wasserstein_distance(A[i],B[j])
                    k.append(np.exp(-dist/sigma))
            K1 += max(k)
        for i in range(n2):
            k= []
            for j in range(n1):
                q1, q2 = Q1[j], Q2[i]

                if q1==q2:
                    dist = wasserstein_distance(A[j],B[i])
                    k.append(np.exp(-dist/sigma))
            K2 += max(k)

        return max([K1,K2])
