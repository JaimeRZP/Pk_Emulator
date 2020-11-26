import scipy
import pyccl as ccl
import numpy as np
import pylab as plt
import itertools

import pymaster as nmt
from numpy import linalg
from sklearn.decomposition import PCA

class LinPkEmulator:
    def __init__(self, n_train, n_k, n_evecs, new = False):
        
        self.n_evecs = n_evecs
        self.k_arr = np.logspace(-4.7,5, n_k)
        self.wc_arr  = np.linspace(0.01, 1, 3*n_train)
        self.h_arr  = np.linspace(0.6,  0.9, n_train)
        
        if new:
            self.grid = []
            for i in range(len(self.wc_arr)):
                for j in range(len(self.h_arr)):
                    self.grid.append(self._get_theory_Pk(self.wc_arr[i], self.h_arr[j]))
            self.grid = np.array(self.grid)
                
            np.savetxt('Pk_grid.txt', self.grid)
        else:
            self.grid = np.loadtxt('Pk_grid.txt')
        
        self.log_grid = np.log(self.grid)
        self.mean_log_grid = np.mean(np.log(self.grid), axis=0) #avg over all cosmologies
        self.clean_grid = self.log_grid - self.mean_log_grid
        
        self.evecs = None
        self.w_arr = None
        self.w_emulator = None
        
        
        
        return 
            
    
    def get_evecs(self):
        if self.evecs is None:
            cov = np.cov(np.transpose(self.clean_grid))
            self.evals, self.evecs = linalg.eig(cov)
            #right to left vectors
            self.evecs = np.transpose(self.evecs)
            self.evecs = self.evecs[:self.n_evecs]
            #back to right
            self.evecs = np.transpose(self.evecs)
            #print(self.evals)        
        return self.evecs

    
    def get_w_arr(self):
        if self.w_arr is None:
            self.w_arr = []
            l=0
            for i in range(len(self.wc_arr)):
                row = []
                for j in range(len(self.h_arr)):
                    #Pk_ij = self._get_theory_Pk(self.wc_arr[i], self.h_arr[j])
                    Pk_ij = self.grid[l]
                    l = l + 1
                    w_ij = self._get_weigths(Pk_ij)
                    row.append(w_ij)
                self.w_arr.append(row)
                
        self.w_arr = np.transpose(np.array(self.w_arr))       
        return self.w_arr
    
    def get_w_emulator(self):
        if self.w_emulator is None:
            self.w_emulator={}
            self.get_w_arr()
            w =0
            for grid in self.w_arr:
                w_i_emulator = scipy.interpolate.interp2d( self.wc_arr, self.h_arr, grid, kind='cubic')
                self.w_emulator["w_{}".format(w)]= w_i_emulator
                w = w+1
        return self.w_emulator
    
    def _get_weigths(self, Pk):
        if self.evecs is None:
            self.get_evecs()
        y = np.log(Pk) - self.mean_log_grid
        return np.linalg.lstsq(self.evecs, y, rcond=None)[0]
        
    def _get_theory_Pk(self, wc, h):
        cosmo = ccl.Cosmology(Omega_c=wc, Omega_b=0.049, h=h, sigma8=0.81, n_s=0.96)
        return ccl.power.linear_matter_power(cosmo, self.k_arr, 1)
        
    def _get_evec_Pk(self, weigths , evecs, mean):
        return mean + np.sum(evecs*weigths, axis=1) 
    
    def get_emulated_Pk(self, h, wc):
        #log lin_Pk
        w_vec = []
        self.get_w_emulator()
        for i in range(self.n_evecs):
            w_i=self.w_emulator["w_{}".format(i)](h, wc)[0]
            w_vec.append(w_i)
        w_vec = np.array(w_vec) 
        Pk = self._get_evec_Pk(w_vec, self.evecs, self.mean_log_grid)
        return self.k_arr, np.array(Pk)
    