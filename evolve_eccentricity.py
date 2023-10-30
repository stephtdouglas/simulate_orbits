"""
Evolve a distribution of binaries in eccentricity following Meibom & Mathieu 2005
"""

import os, sys

import numpy as np
import astropy.units as u
from astropy.time import Time

import twobody

def ecc_ode(pars,t):

    Aconst = 1

    ee, PP = pars

    de_dt = -1*ee / (A*PP**(16/3))
    dP_dt = -3*ee**2 / (A*PP**(13/3))

    return [de_et,dP_dt]
