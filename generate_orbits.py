
import os, sys

import numpy as np
import astropy.units as u
from astropy.time import Time

import twobody

rnd = np.random.default_rng(seed=42)


def ecc_init(n_stars=10,e_avg=0.35,e_sigma=0.15,P_lims=[0.01,100]):
    """
    Produce a random set of n_stars binaries, normally distributed in eccentricity
    space (set by e_avg and e_sigma) and uniformly distributed in period space 
    between P_lims.
    """

    eccentricities = rnd.normal(loc=e_avg,scale=e_sigma,size=n_stars)
    eccentricities[eccentricities<0] = 0
    eccentricities[eccentricities>1] = 1
    periods = rnd.uniform(low=P_lims[0],high=P_lims[1],size=n_stars)*u.day

    return eccentricities, periods


def orbit_init(n_stars=10,**ecc_kwargs):

    eccentricities, periods = ecc_init(n_stars,**ecc_kwargs)

    omega = rnd.uniform(low=0,high=2*np.pi,size=n_stars)*u.radian # argument of pericenter
    inc = rnd.uniform(low=0,high=np.pi,size=n_stars)*u.radian # inclination

    M0 = rnd.uniform(low=0,high=2*np.pi,size=n_stars)*u.radian # phase at reference time

    # Omega is the Longitude of ascending node; no impact on RVs
    # we'll set t0 to 2015.0 for everything as well

    K = rnd.uniform(low=1,high=100,size=n_stars)*u.km/u.s

    all_elements = []
    for i in range(n_stars):
        all_elements.append(twobody.KeplerElements(P=periods[i],e=eccentricities[i],
                                                   K=K[i],omega=omega[i],i=inc[i],
                                                   Omega=0*u.degree,t0=Time('J2015.0'),
                                                   M0=M0[i]))

    return all_elements

def generate_rvs(elements_list,output_filename,times=None):
    """
    Generate fake RV datasets from a list of orbital elements
    """

    if times==None:
        rand_times = rnd.uniform(low=0,high=12,size=18)*u.year
        t = Time('2007-01-10') + rand_times 
    max_obs = len(t)
    time_idx = np.arange(max_obs)

    with open(output_filename,"w") as f:
        f.write("star,JD,rv(km/s)\n")

        for i,elem in enumerate(elements_list):
            orb = twobody.KeplerOrbit(elem)
            rv = orb.radial_velocity(t)

            n_obs = rnd.integers(low=3,high=max_obs)
            obs_j = rnd.choice(time_idx,size=n_obs)

            for j in obs_j:
                f.write(f"SYN{i:05d},{t[j].jd:.6f}")
                f.write(f",{rv[j].to(u.km/u.s).value:.2f}\n")

if __name__=="__main__":

    n_stars=10

    elem_list = orbit_init(n_stars)

    generate_rvs(elem_list,"test_syn_out.csv")