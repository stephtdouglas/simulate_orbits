
import os, sys

import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt

import twobody

from models import circ_function

rnd = np.random.default_rng(seed=42)

# a_avg_FGK = 4.5 # AU
# m_tot = (0.8+1) # Msun
# P_avg_yr = np.sqrt(a_avg_FGK**3/m_tot) # Year
# P_avg = (P_avg_yr*u.year).to(u.day)
# log_P_avg = np.log10(P_avg.value)


def ecc_init(n_stars=10,e_avg=0.35,e_sigma=0.15,
             log_P_avg=5,log_P_sigma=0.5,
             P_lims=[0.1,1000],plot_filename=None):
    """
    Produce a random set of n_stars binaries, 
    normally distributed in eccentricity space (set by e_avg and e_sigma) 
    and uniformly distributted in eccentricity space.

    Also possible to run with 
    normally distributed in period space (set by log_P_avg and 
    log_P_sigma in log10(days )

    Default period distribution taken from Duchene & Kraus (2013), Sec 3.1.2
    """

    eccentricities = rnd.normal(loc=e_avg,scale=e_sigma,size=n_stars)
    eccentricities[eccentricities<0] = 0
    eccentricities[eccentricities>1] = 1

    # log_periods = rnd.normal(loc=log_P_avg,scale=log_P_sigma,size=n_stars)
    # periods = (10**log_periods) * u.day

    log_periods = rnd.uniform(low=np.log10(P_lims[0]),high=np.log10(P_lims[1]),
                              size=n_stars)
    periods = (10**log_periods)*u.day
    # periods = rnd.uniform(low=P_lims[0],high=P_lims[1],size=n_stars)*u.day

    if plot_filename is not None:
        plt.plot(periods,eccentricities,'o')
        plt.xlim(P_lims[0],P_lims[1])
        plt.xscale("log")
        plt.ylim(-0.01,1)
        plt.savefig(plot_filename,facecolor="w")

    return eccentricities, periods


def ecc_circularized(n_stars,Pcirc=7*u.day,plot_filename=None,**ecc_kwargs):
    """
    Produce a random set of n_stars binaries, where eccentricities are
    determined following the circularization function from Meibom & Mathieu 2005

    TODO: right now it just cuts the distribution at Pcirc because I don't have time
    """

    raw_eccentricities, periods = ecc_init(n_stars,**ecc_kwargs)

    x_periods = np.logspace(-1,4,100)
    y_ecc = circ_function(x_periods,Pcirc.to(u.day).value)

    eccentricities = raw_eccentricities
    eccentricities[periods<Pcirc] = 0    

    if plot_filename is not None:
        plt.plot(periods,eccentricities,'o')
        plt.plot(x_periods,y_ecc,'-')
        # plt.xlim(P_lims[0],P_lims[1])
        plt.xscale("log")
        plt.ylim(-0.01,1)
        plt.savefig(plot_filename,facecolor="w")

    return eccentricities, periods


def orbit_init(n_stars=10,q_gamma=4.2,output_filename=None,circ=False,**ecc_kwargs):

    if circ:
        eccentricities, periods = ecc_circularized(n_stars,**ecc_kwargs)
    else:
        eccentricities, periods = ecc_init(n_stars,**ecc_kwargs)

    omega = rnd.uniform(low=0,high=2*np.pi,size=n_stars)*u.radian # argument of pericenter
    inc = rnd.uniform(low=0,high=np.pi,size=n_stars)*u.radian # inclination

    M0 = rnd.uniform(low=0,high=2*np.pi,size=n_stars)*u.radian # phase at reference time

    # Omega is the Longitude of ascending node; no impact on RVs
    # we'll set t0 to 2015.0 for everything as well

    q = rnd.power(q_gamma+1,size=n_stars)

    M1 = 1*u.solMass
    M2 = M1*q

    aa = (M2.value * periods.to(u.year).value**2)**(1/3) * u.au

    # K = rnd.uniform(low=1,high=100,size=n_stars)*u.km/u.s

    all_elements = []
    for i in range(n_stars):
        all_elements.append(twobody.KeplerElements(P=periods[i],e=eccentricities[i],
                                                   a=aa[i],omega=omega[i],i=inc[i],
                                                   Omega=0*u.degree,t0=Time('J2015.0'),
                                                   M0=M0[i]))

    if output_filename is not None:
        with open(output_filename,"w") as f:
            f.write("P(year),P(day),q,M2,a(au),omega(rad),i(rad),M0(rad),Omega,T0\n")
            for i in range(n_stars):
                f.write(f"{periods[i].to(u.year).value:.6f},")
                f.write(f"{periods[i].to(u.day).value:.6f},")
                f.write(f"{q[i]:.2f},")
                f.write(f"{M2[i].to(u.solMass).value:.2f},")
                f.write(f"{aa[i].to(u.au).value:.2f},")
                f.write(f"{omega[i].to(u.radian).value:.3f},")
                f.write(f"{inc[i].to(u.radian).value:.3f},")
                f.write(f"{M0[i].to(u.radian).value:.3f},")
                f.write(f"0.0,J2015.0\n")

    return all_elements

def generate_rvs(elements_list,output_filename,times=None):
    """
    Generate fake RV datasets from a list of orbital elements
    """

    if times==None:
        rand_times = rnd.uniform(low=0,high=12,size=18)*u.year
        t = Time('2007-01-10') + rand_times 
    elif times=="NGC 6811":
        dates = []
        with open(os.path.expanduser("~/Dropbox/data/hecto_date_list.txt"),"r") as g:
            l = g.readline()
            while ("ngc6811" in l) is False:
                l = g.readline()
            l = g.readline()
            l = g.readline()
            while (l!="\n"):
                dates.append(l.strip())
                l = g.readline()
                if l=="":
                    break

        # print(dates)
        t = Time(dates,format="decimalyear")

        
    max_obs = len(t)
    time_idx = np.arange(max_obs)

    with open(output_filename,"w") as f:
        f.write("star,JD,rv(km/s)\n")

        for i,elem in enumerate(elements_list):
            orb = twobody.KeplerOrbit(elem)
            rv = orb.radial_velocity(t)

            if elem.K>(20*u.km/u.s):
                low_obs = 8
            else:
                low_obs = 3
            n_obs = rnd.integers(low=low_obs,high=max_obs)
            obs_j = rnd.choice(time_idx,size=n_obs)

            for j in obs_j:
                f.write(f"SYN{i:05d},{t[j].jd:.6f}")
                f.write(f",{rv[j].to(u.km/u.s).value:.2f}\n")

if __name__=="__main__":

    n_stars=100

    # ee,pp = ecc_init(n_stars,plot_filename="test_syn_orbits.png",P_lims=[0.1,1e4])
    # plt.close()
    # ee,pp = ecc_circularized(n_stars,plot_filename="test_syn_orbits_circ.png",P_lims=[0.1,1e4])
    # plt.close()

    elem_list = orbit_init(n_stars,output_filename="test_syn_orbits.csv",
                           circ=False,plot_filename="test_syn.png")
    generate_rvs(elem_list,output_filename="test_syn_rvs.csv",times="NGC 6811")
    plt.close()

    elem_list_circ = orbit_init(n_stars,output_filename="test_syn_circ_orbits.csv",
                                circ=True,plot_filename="test_syn_circ.png")
    generate_rvs(elem_list_circ,output_filename="test_syn_circ_rvs.csv",times="NGC 6811")
    plt.close()
