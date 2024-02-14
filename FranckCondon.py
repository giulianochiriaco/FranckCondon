import numpy as np
from scipy.special import *
import matplotlib.pyplot as plt

def Vmorse(R,De,Re,we):
    return De*(np.exp(-2*(R-Re)/we)-2*np.exp(-(R-Re)/we))

def omega_nu(lambd):
    "Gives the vector of the energies of the bound vibrational states"
    nuv = np.arange(int(lambd-0.5))
    return -(lambd-nuv-0.5)**2/lambd**2
    
def psiR(R,lambd,nu,Re,we,norm=0):
    "Gives the eigenfunction of mode nu as function of radial coordinate R for a Morse potentialgiven by Re and we"
    "lambd is sqrt(2*m*De)*we/hbar"
    zv=2*lambd*np.exp(-(R-Re)/we)
    "z is the coordinate in which the Schroedinger equation is solver"
    psi = psiz(zv,lambd,nu)
    N = np.sqrt(np.sum(np.conjugate(psi)*psi))
    if norm==0:
        return psi#/np.sqrt(we)
    else:
        return psi/N

def psiz(z,lambd,nu):
    "Gives the NON-normalized wavefunction of mode nu as function of the new coordinate z"
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


def AfcDipole(lambd1,Re1,we1,lambd2,Re2,we2):
    "Gives a matrix (table) with the dipole matrix elements between the vibrational levels of two different Morse potentials"
    "The Morse potentials are characterized by different lambda (i.e. De), Re and we"
    "Useful to calculate molecular-dipole induced transitions"
    nuM1=int(np.floor(lambd1-0.5)) #find max number of levels for Morse 1"
    nuM2=int(np.floor(lambd2-0.5)) #find max number of levels for Morse 2"
    Amunu=np.zeros((nuM1+1,nuM2+1))
    Rv = np.linspace(0,50*max(we1,we2)+max(Re1,Re2),10000) #define vector of coordinates"
    for nu1 in range(nuM1+1):
        psi1 = psiR(Rv,lambd1,nu1,Re1,we1) #get eigenfunction 1"
        N1 = np.sum(psi1*np.conjugate(psi1)) #and its normalization"
        for nu2 in range(nuM2+1):
            psi2 = psiR(Rv,lambd2,nu2,Re2,we2) #"get eigenfunction 2"
            N2 = np.sum(psi2*np.conjugate(psi2)) #"and its normalization"
            Amunu[nu1,nu2] = np.sum(psi1*Rv*psi2)/np.sqrt(N1*N2) #"compute overlap"
    return Amunu

def psiE(R,E,lambd,Re,we):
    "Numerically solve the Schroedinger equation for unbound states with energy E (measured in units of De)"
    "The parameters lambd, Re and we characterize the Morse potential"
    "First define a suitable range of coordinates R"
    Ns = 10000
    Rv = np.linspace(0,50*we+Re,Ns)
    h = (50*we+Re)/Ns
    "Then define the vector k(r) for the Numerov method. Use the Morse potential with right parameters and De=1"
    VMv = Vmorse(Rv,1,Re,we)
    kv = np.sqrt(E-np.array(VMv,dtype=complex))*lambd/we
    "Then impose the boundary conditions on the wavefunction at large R"
    psiv1 = np.zeros(Ns,dtype=complex)
    psiv1[Ns-1] = 1
    psiv1[Ns-2] = 1-1j*h*kv[Ns-1]
    psiv2 = np.zeros(Ns,dtype=complex)
    psiv2[Ns-1] = 1
    psiv2[Ns-2] = 1+1j*h*kv[Ns-1]
    for i in np.arange(0,Ns-2):
        psiv1[Ns-3-i] = (2*psiv1[Ns-i-2]*(1-5*h**2*kv[Ns-i-2]**2/12)-psiv1[Ns-i-1]*(1+h**2*kv[Ns-i-1]**2/12))/(1+h**2*kv[Ns-i-3]**2/12)
        psiv2[Ns-3-i] = (2*psiv2[Ns-i-2]*(1-5*h**2*kv[Ns-i-2]**2/12)-psiv2[Ns-i-1]*(1+h**2*kv[Ns-i-1]**2/12))/(1+h**2*kv[Ns-i-3]**2/12)
    return psiv1,psiv2,kv


def psiE2(R,E,lambd,Re,we):
    "Numerically solve the Schroedinger equation for unbound states with energy E (measured in units of De)"
    "The parameters lambd, Re and we characterize the Morse potential"
    "First define a suitable range of coordinates R"
    Ns = 10000
    Rv = np.linspace(0,20*we+Re,Ns)
    h = (50*we+Re)/Ns
    "Then define the vector k(r) for the Numerov method. Use the Morse potential with right parameters and De=1"
    VMv = Vmorse(Rv,1,Re,we)
    kv = np.sqrt(E-np.array(VMv,dtype=complex))*lambd/we
    "Then impose the boundary conditions on the wavefunction at large R"
    psiv1 = np.zeros(Ns,dtype=complex)
    psiv1[0] = np.exp(1j*kv[0])
    psiv1[1] = np.exp(1j*kv[1])
    for i in np.arange(1,Ns-1):
        psiv1[i+1] = (2*psiv1[i]*(1-5*h**2*kv[i]**2/12)-psiv1[i-1]*(1+h**2*kv[i-1]**2/12))/(1+h**2*kv[i+1]**2/12)
    return psiv1/np.max(psiv1),kv,Rv

def Ec_filter(Et,tv,r,wc):
    "An input electric field Et (in real time) filtered by the cavity"
    Nt = len(tv)
    Nr = -int(10/np.log(r))
    dt = tv[1]-tv[0]
    dn = np.pi/(dt*wc)
    out = np.zeros(Nt,dtype=complex)
    for n in np.arange(Nr):
        Eshift = np.zeros(Nt,dtype=complex)
        Dn = int(dn*n)
        if Dn<Nt:
            Eshift[Dn:] = Et[:Nt-Dn]
            out = out +(-r)**n*Eshift
    return out
    
def Ec_Gaussian(tv,wL,tau,wc,r):
    "An input Gaussian pulse with main frequency wL and pulse duration tau filtered by the cavity"
    Nr = -int(10/np.log(r))
    out = np.zeros(len(tv),dtype=complex)
    for n in range(Nr):
        out = out + (-r)**n*np.exp(-1j*wL*(tv-n*np.pi/wc))*np.exp(-(tv-n*np.pi/wc)**2/tau**2)
    return out

def evol_matr(tvec,El,omega_mu,omega_nu,Amunu,gmunu):
    "Attempt at constructing the matrix of the evolution equations for the populations and coherences. not complete yet"
    N_mu = len(omega_mu)
    N_nu = len(omega_nu)
    size = (N_mu+1)*(N_nu+1)-1
    Nt = len(tvec)
    matr = np.zeros((Nt,size,size))
    for t in range(Nt):
        OmR = Amunu*np.exp(1j*(np.outer(vmu,np.ones(N_nu))-np.outer(np.ones(N_mu),vnu)*tvec[t]))*El[t]
        matr[t,:N_mu,N_mu:N_mu+N_nu] = gmunu
        matr[t,N_mu:N_mu+N_nu,N_mu:N_mu+N_nu] = -np.diag(np.sum(gmunu,0))
        matr[t,N_mu+N_nu:,N_mu+N_nu:] = -np.diag(np.sum(gmunu,0),np.ones(N_mu))/2
    return matr
