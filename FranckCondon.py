import numpy as np
from scipy.special import *
import matplotlib.pyplot as plt

def Vmorse(R,De,Re,we):
    return De*(np.exp(-2*(R-Re)/we)-2*np.exp(-(R-Re)/we))

def psiR(R,lambd,nu,Re,we):
    zv=2*lambd*np.exp(-(R-Re)/we)
    return psiz(zv,lambd,nu)#/np.sqrt(we)

def psiz(z,lambd,nu):
    N=np.sqrt((2*lambd-2*nu-1)*factorial(nu)/gamma(2*lambd-nu))
    return np.exp(-z/2)*genlaguerre(nu,2*lambd-2*nu-1)(z)*z**(lambd-nu-1/2)#*N

def Afc(lambd1,Re1,we1,lambd2,Re2,we2):
    nuM1=int(np.floor(lambd1-0.5))
    nuM2=int(np.floor(lambd2-0.5))
    Amunu=np.zeros((nuM1+1,nuM2+1))
    Rv = np.linspace(0,50*max(we1,we2)+max(Re1,Re2),10000)
    for nu1 in range(nuM1+1):
        psi1 = psiR(Rv,lambd1,nu1,Re1,we1)
        N1 = np.sum(psi1*np.conjugate(psi1))
        for nu2 in range(nuM2+1):
            psi2 = psiR(Rv,lambd2,nu2,Re2,we2)
            N2 = np.sum(psi2*np.conjugate(psi2))
            Amunu[nu1,nu2] = np.sum(psi1*psi2)/np.sqrt(N1*N2)
    return Amunu

