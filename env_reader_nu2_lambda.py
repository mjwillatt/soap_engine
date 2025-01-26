#!/usr/bin/python3

import re
import itertools
import numpy as np
from ase.geometry import find_mic
from sympy.physics.wigner import wigner_3j
from ge import gaussian_expansion as ge

########################################################################################## 

def cutoff(r, rcut):
  '''
  Smooth cutoff function based on raised cosine
  r = distance
  rcut = cutoff radius
  '''
  #width of raised cosine
  ctwidth = 0.5
  if r <= rcut - ctwidth: cutoff = 1.0
  elif r >= rcut: cutoff = 0.0
  else: cutoff = 0.5*(1.0 + np.cos(np.pi*(r - rcut + ctwidth)/ctwidth))
  return cutoff

########################################################################################## 

def density(nmax, lmax, brcut, sigma):
  '''
  Outer part of closure for expansion of Gaussian density on a radial and angular basis
  nmax = number of radial basis functions (Gauss-Legendre quadrature) - 1
  lmax = number of angular basis functions (spherical harmonics) - 1
  brcut = endpoint of Gauss-Legendre grid
  sigma = Gaussian width
  '''
  #Gauss-Legendre quadrature points and weights
  x, w = ge.gaulegf(x1=0.0, x2=brcut, n=nmax+1)

  def inner(rij, cost, phi, lmax=lmax):
    '''
    Inner part of closure for expansion of Gaussian density on a radial and angular basis
    rij = distance of Gaussian from origin
    cost = cos of theta (polar angle)
    phi = azimuthal angle
    lmax = number of angular basis functions (spherical harmonics)
    '''
    #radial expansion for each l
    #only need to know the distance of the Gaussian from origin
    f1 = np.zeros((nmax+1, lmax+1))
    for i in range(nmax+1):
      #quad precision required for numerical stability
      f1[i, :] = ge.fsubquad(r2=x[i], sigma2=sigma, rij2=rij, lmax=lmax)
      f1[i, :] *= np.sqrt(w[i])
    #extend the expansion to m >= 0
    #now we need knowledge of the position of the Gaussian (cost and phi)
    f2 = np.zeros((nmax+1, lmax+1, lmax+1))
    f2 = ge.f2sub(f1, cost=cost)
    #factor of sqrt(2) required because we are ignoring m < 0
    f2[:, :, 1:] *= np.sqrt(2.0)
    eiphi = np.cos(phi) + 1.0j*np.sin(phi)
    eiphiar = np.array([eiphi**i for i in range(lmax+1)])
    f2 = f2.astype(complex)
    f2 *= eiphiar
    return f2

  return inner

##########################################################################################

def rconvert(r):
  '''
  Function to go from Cartesian to spherical polar coordinates
  r = Cartesian displacement (3D NumPy array)
  '''
  #rij = distance
  #cost = cost of theta (polar angle)
  #phi = azimuthal angle
  rij = np.sqrt(np.dot(r, r))
  if rij > 0.0:
    cost = r[2]/rij
  else:
    cost = 1.0
  phi = np.arctan2(r[1], r[0])
  return rij, cost, phi

##########################################################################################

def nu3_descriptor(nmax, lmax, lam, inv=0):
  '''
  Outer part of closure for calculating the three-body lambda SOAP feature vector
  Closure required for encapsulating dictionary w3j (lookup table)
  Also calculate and save indices of non-zero feature vector components
  nmax = number of radial basis functions (Gauss-Legendre quadrature) - 1
  lmax = number of angular basis functions (spherical harmonics) - 1
  lam = lambda, i.e. l of the spherical harmonic the feature vector transforms as
  inv = binary variable for inversion symmetrization
  '''
  relevant_indices = {}
  w3j = {}
  for j1 in range(lmax+1):
    for j2 in range(lmax+1):
      if (j1 + j2 + lam) % 2 == inv: continue
      relevant_indices[(j1, j2, lam)] = []
      for m1 in range(2*j1+1):
        for m2 in range(2*j2+1):
          for m3 in range(2*lam+1):
            #shift m from the range 0 <= m <= 2*l + 1 to -l <= ms <= l
            m1s = m1 - j1
            m2s = m2 - j2
            m3s = m3 - lam
            #If Wigner 3j symbol is finite
            #Save the indices in relevant_indices
            #Save the values in w3j
            if float(wigner_3j(j1, j2, lam, m1s, m2s, m3s)) != 0.0:
              relevant_indices[(j1, j2, lam)] += \
                  [(m1s+lmax, m2s+lmax, m3s+lam)]
              w3j[(j1, j2, lam, m1s+lmax, m2s+lmax, m3s+lam)] = \
                  float(wigner_3j(j1, j2, lam, m1s, m2s, m3s))
      if relevant_indices[(j1, j2, lam)] == []: 
        relevant_indices.pop((j1, j2, lam), None)
  #lenri = number of non-zero feature vecture components
  lenri = len(relevant_indices.keys())

  def inner(f):
    '''
    Inner part of closure for calculating the three-body lambda SOAP feature vector
    f = expansion of the gaussian density (NumPy array of shape (nmax+1, lmax+1, lmax+1))
    '''
    n1 = f.shape[0]
    descriptor = np.zeros((n1**2, lenri, 2*lam+1), dtype=complex)
    for i, (j1, j2, j3) in enumerate(relevant_indices.keys()):
      nu3 = np.zeros((n1**2, 2*lam+1), dtype=complex)
      for j, (m1, m2, m3) in enumerate(relevant_indices[(j1, j2, j3)]):
        nu3[:, m3] += w3j[(j1, j2, j3, m1, m2, m3)]* \
                      np.einsum('j,k -> jk', f[:, j1, m1], f[:, j2, m2]).ravel()
      for m3 in range(2*lam + 1):
        descriptor[:, i, m3] = nu3[:, m3]
    descriptor = descriptor.reshape((n1, n1, lenri, 2*lam+1))
    return descriptor

  return lenri, inner

##########################################################################################
 
def get_nu3_descriptor_pairs(xyz, species1, species2, rcut, nmax, lmax, gdens, nu3d, 
                             lenri, lweight, lam):
  '''
  Compute feature vectors for a series of molecules
  xyz = ase Atoms object for the molecule
  species1 = string describing species and label of atom 1, e.g. C2
  species2 = string describing species and label of atom 2, e.g. C2
  rcut = cutoff radius
  nmax = number of radial basis functions (Gauss-Legendre quadrature) - 1
  lmax = number of angular basis functions (spherical harmonics) - 1
  gdens = function for expansion of gaussian density
  nu3d = function for calculating the three-body lambda SOAP feature vector
  lenri = number of relevant (angular) indices in feature vector
  lweight = weight for floating centre: (lweight*r1 + r2)/(1 + lweight)
  lam = lambda, i.e. l of the spherical harmonic the feature vector transforms as
  '''
  coords = xyz.get_positions()
  ans = xyz.get_atomic_numbers()
  syms = xyz.get_chemical_symbols()
  nspecies = len(set(syms))

  #decompose string species1 into sym1 (chemical symbol) and id1 (atom label)
  sym1 = re.match('([A-Z]*)[0-9]*', species1).group(1)
  sym2 = re.match('([A-Z]*)[0-9]*', species2).group(1)
  id1 = int(re.match('[A-Z]*([0-9]*)', species1).group(1))
  id2 = int(re.match('[A-Z]*([0-9]*)', species2).group(1))

  #get the indices of atoms 1 and 2
  centind1 = [i for i, j in enumerate(syms) if j == sym1][id1-1]
  centind2 = [i for i, j in enumerate(syms) if j == sym2][id2-1]
  centre = (centind1, centind2)

  #determine if we have periodic boundary conditions
  cell = xyz.get_cell()
  if cell.sum() == 0.0: pbc = False
  else: pbc = xyz.get_pbc()

  #get ready to calculate density expansions
  f = np.zeros((nspecies, nmax+1, lmax+1, lmax+1), dtype=complex) 
  f2 = np.zeros((nspecies, nmax+1, lmax+1, 2*lmax+1), dtype=complex)
  #csphase = Condon-Shortley phase for spherical harmonics
  csphase = np.array([(-1)**m for m in range(lmax+1)])[::-1]

  #calculate the coordinates of the floating centre
  #find displacements of all atoms from the floating centre
  mpoint = (lweight*coords[centre[0]] + coords[centre[1]])/(1.0 + lweight)
  dr = find_mic(coords - mpoint, cell=cell, pbc=pbc)[0]
  dr1 = find_mic(coords - coords[centre[0]], cell=cell, pbc=pbc)[0]
  dr2 = find_mic(coords - coords[centre[1]], cell=cell, pbc=pbc)[0]

  #calculate density expansions, one for each chemical element in the molecule
  for i, spec in enumerate(list(set(ans))):
    labels = np.where(ans == spec)[0]
    for j in labels:
      rij, cost, phi = rconvert(dr[j])
      r1 = rconvert(dr1[j])[0]
      r2 = rconvert(dr2[j])[0]
      if r1 >= rcut or r2 >= rcut: continue
      f[i] += gdens(rij, cost, phi)*np.sqrt(cutoff(r1, rcut)*cutoff(r2, rcut))

  #fill in the missing entries of the expansion corresponding to m < 0
  f[:, :, :, 1:] /= np.sqrt(2.0)
  f2[:, :, :, lmax+1:] = f[:, :, :, 1:]
  f2[:, :, :, 0:lmax+1] = csphase*np.conj(f[:, :, :, ::-1])
  f2 = f2.reshape((nspecies*(nmax+1), lmax+1, 2*lmax+1))

  #calculate the descriptors
  desc = nu3d(f2)
  desc = desc.reshape((-1, 2*lam+1))
  desc = desc.transpose()
  #apply a complex to real transformation so that the lambda spherical harmonic is real
  desc = np.matmul(comp_to_real(lam), desc)

  #compress the feature vector by making it real and accounting for symmetry
  desc2 = np.zeros(desc.shape)
  desc2 = np.real(desc) + np.imag(desc)
  desc2 = desc2.reshape((2*lam+1, nspecies*(nmax+1), nspecies*(nmax+1), lenri))
  desc = np.zeros((2*lam+1, int(nspecies*(nmax+1)*(nspecies*(nmax+1)+1)/2), lenri))
  counter = 0
  for i in range(nspecies*(nmax+1)):
    for j in range(i, nspecies*(nmax+1)):
      if i == j: mult = 1.0
      else: mult = np.sqrt(2.0)
      desc[:, counter, :] = mult*desc2[:, i, j, :]
      desc[:, counter, :] = mult*desc2[:, i, j, :]
      counter += 1

  desc = desc.reshape((2*lam+1, -1))
  #return a list of feature vectors, each corresponding to single value of mu
  desclist = [desc[i] for i in range(2*lam+1)]
  return desclist

##########################################################################################

def comp_to_real(l):
  '''
  Calculate the complex to real transformation matrix for spherical harmonics
  l = spherical harmonic l (azimuthal quantum number)
  '''
  ctore = np.zeros((2*l+1, 2*l+1), dtype=complex)
  for m in range(1, l+1):
    #m < 0
    ctore[l+m, l-m] = 1.0
    ctore[l-m, l-m] = -1.0j
    #m > 0
    ctore[l+m, l+m] = (-1.0)**m
    ctore[l-m, l+m] = (-1.0)**m*1.0j
  ctore /= np.sqrt(2.0)
  #m = 0
  ctore[l, l] = 1.0
  return ctore

##########################################################################################
