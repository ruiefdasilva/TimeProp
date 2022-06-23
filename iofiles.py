###############################################
# Code to solve the TDSE in 1D Hubbard Model
# Author: Rui Silva
# Version 1.0
# Module for writing and reading files
###############################################

import numpy as np
import scipy as sp
from scipy import sparse as sp_sparse
from scipy.sparse import linalg as sp_sparse_linalg
import sys
from subprocess import call
###############################################



def WriteFile(list_arrays,title,filename):
    N_arrays=len(list_arrays)
    size_array=len(list_arrays[0])
    for i in range(N_arrays):
        if list_arrays[i].ndim!=1:
            print('Error in IOfile 1')
            return 0
        if len(list_arrays[i])!=size_array:
            print('Error in IOfile 2')
            return 0
    BigArraySave=np.zeros((size_array,N_arrays))
    for i in range(N_arrays):
        BigArraySave[:,i]=list_arrays[i]
    np.savetxt(filename,BigArraySave, fmt='%.30e', delimiter='\t', newline='\n', header=' '+title, footer='', comments='# ')
    return 1


