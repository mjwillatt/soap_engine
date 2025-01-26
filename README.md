SOAP Engine
===========

SOAP Engine is an implementation of Smooth Overlap of Atomic Positions (SOAP)
for tensor properties, i.e. lambda SOAP. It is designed specifically for
Hamiltonian learning. The computationally demanding task of expanding a
Gaussian on a basis (radial and angular) is handled by Fortran with Numerical
Recipes routines. Everything else is handled by Python 3. f2py is used to
integrate the two.

gaussian_expansion.f90
----------------------

Contains routines for expanding a Gaussian on spherical harmonics and Gauss-Legendre
radial grid points.

env_reader_nu2_lambda.py
------------------------

Contains routines for calculating tensor SOAP components from the Gaussian expansion.

get_soaps_nu2_lambda.py
-----------------------

Calculates tensor SOAP components for a collection of atomic structures,
provided in an xyz file. Designed for Hamiltonian learning, where one sometimes
needs to centre the basis away from atoms. Providing as inputs -species1 "H2"
-species2 "O1" --lweight 2 centres the basis at (2*r1 + r2)/(2 + 1), where r1
and r2 are respectively the positions of the second hydrogen and first oxygen.
The ordering, i.e. "second", is the same as in the xyz file.  Depending on the
arguments --inv and --lam, only some spherical harmonic l1, l2 combinations are
kept; if (l1 + l2 + lam) % 2 == inv, the component is discarded. This is needed
to ensure the descriptors transform correctly under structure inversion.
