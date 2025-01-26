#!/usr/bin/python3 -u

import argparse
import numpy as np
import env_reader_nu2_lambda as er
from ge import gaussian_expansion as ge
from ase.io import read 

##########################################################################################

def get_soaps(species1, species2, rc, nmax, lmax, gdens, lam, inv, lweight):
  '''
  Outer part of closure for combining the routines in env_reader_nu2_lambda
  species1 = string describing species and label of atom 1, e.g. C2
  species2 = string describing species and label of atom 2, e.g. C2
  rc = cutoff radius
  nmax = number of radial basis functions (Gauss-Legendre quadrature) - 1
  lmax = number of angular basis functions (spherical harmonics) - 1
  gdens = function for expansion of gaussian density
  lam = lambda, i.e. l of the spherical harmonic the feature vector transforms as
  inv = binary variable for inversion: 0 for odd rule; 1 for even rule
  lweight = floating centre offset weight
  '''
  lenri, nu3d = er.nu3_descriptor(nmax, lmax, lam, inv=inv)

  def inner(frames):
    '''
    Inner part of closure for combining the routines in env_reader_nu2_lambda
    Returns a list of feature vectors using the parameters defined in the outer part
    frames = list of ase Atoms objects
    '''
    n = len(frames)
    soaps_list = []
    for i in range(n):
      soap = er.get_nu3_descriptor_pairs(frames[i], species1, species2, rc, nmax, lmax, 
                                         gdens, nu3d, lenri, lweight, lam)
      soap = np.vstack(soap)
      soaps_list += [soap]
    #soaps_list is a list of feature vectors, each element corresponding to a molecule
    return soaps_list

  return inner

##########################################################################################
##########################################################################################

def main(fxyz, species1, species2, rc, nmax, lmax, awidth, nframes, inv, lweight, lam):
  '''
  Generates lambda SOAP feature vectors for molecules in xyz file
  fxyz = location of xyz file
  species1 = string for first species, e.g. "H2"
  species2 = string for second species, e.g. "O1"
  rc = cutoff radius
  nmax = maximum radial label
  lmax = maximum angular label
  awidth = atom width (width of Gaussians)
  nframes = number of frames (molecules) to read from xyz file
  inv = binary variable for inversion: 0 for odd rule; 1 for even rule
  lweight = floating centre offset weight
  lam = lambda for lambda-SOAP
  '''
  fxyz = str(fxyz)
  species1 = str(species1)
  species2 = str(species2)
  nmax = int(nmax)
  lmax = int(lmax)
  awidth = float(awidth)
  nframes = int(nframes)
  inv = int(inv)
  lweight = float(lweight)
  lam = int(lam)
  #if nframes == 0 read every molecule in xyz file
  if nframes == 0: nframes = ''
  frames = read(fxyz, ':'+str(nframes))
  nframes = len(frames)
  brcut = rc + 3.0*awidth
  gdens = er.density(nmax, lmax, brcut, awidth)
  gsoaps = get_soaps(species1, species2, rc, nmax, lmax, gdens, lam, inv, lweight)
  soaps = gsoaps(frames)
  #soaps is a list of feature vectors, each element corresponding to a molecule
  return soaps
  
##########################################################################################
##########################################################################################

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-fxyz', type=str, help='Location of xyz file')
  parser.add_argument('-species1', type=str, help='String for first species, e.g. "H2"')
  parser.add_argument('-species2', type=str, help='String for second species, e.g. "O1"')
  parser.add_argument('--rc', type=float, default=3.0, help='Cutoff radius')
  parser.add_argument('--nmax', type=int, default=9, help='Maximum radial label')
  parser.add_argument('--lmax', type=int, default=9, help='Maximum angular label')
  parser.add_argument('--awidth', type=float, default=0.3, help='Atom width')
  parser.add_argument('--nframes', type=int, default=0, help='Number of frames')
  parser.add_argument('--inv', type=int, default=0, help='0 for odd rule; 1 for even rule')
  parser.add_argument('--lweight', type=float, default=1.0, help='Centre-offset weight')
  parser.add_argument('--lam', type=int, default=0, help='Lambda for lambda SOAP')
  args = parser.parse_args()

  main(args.fxyz, args.species1, args.species2, args.rc, args.nmax, args.lmax, \
       args.awidth, args.nframes, args.inv, args.lweight, args.lam)
