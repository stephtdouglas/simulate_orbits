import os, sys, datetime

import matplotlib.pyplot as plt
import numpy as np
import astropy.stats as stats
import astropy.io.ascii as at
import astropy.units as u
import astropy.units.cds as cds
import corner
import pymc as pm
#import pymc3_ext as pmx
#import exoplanet as xo
import arviz as az

import schwimmbad

import thejoker as tj
from thejoker.samples_analysis import is_P_unimodal, is_P_Kmodal

# set up a random generator to ensure reproducibility
#rnd = np.random.default_rng(seed=42)


#hostname = os.getenv("HOSTNAME",default="hpc.cluster")
#if hostname=="hpc.cluster":
cache_dir = "/data2/labs/douglaslab/douglste/orbits/jokercache/"
results_dir = "/data2/labs/douglaslab/douglste/orbits/jokerresults/"
#else:
#    cache_dir = os.path.expanduser("~/projects/jokercache/")
#    results_dir = os.path.expanduser("~/projects/jokerresults/")

def clown_car(data,name,pmin,pmax,nsamples=int(7e4),jitter=False,
              to_plot=False,prior_cache_file=None,mpi=False):
    """
    Run a preliminary Joker rejections sampling test on a target

    data: RVData object

    name: string giving the object name/identifier

    pmin, pmax: min and max periods to test

    nsamples: number of samples to try

    jitter: whether to include unspecified jitter

    to_plot: whether to plot and save the RV data before fitting
    """

    if to_plot is True:
        ax = data.plot()
        ax.set_xlabel("Time [JD]")
        ax.set_ylabel("RV [km/s]")
        print(os.path.exists(os.path.join(results_dir,"plots/")))
        plt.savefig(os.path.join(results_dir,"plots/{0}_data.png".format(name)))
        plt.close()

    if jitter is True:
        print("Jitter is no longer enabled")
    prior = tj.JokerPrior.default(P_min=pmin, P_max=pmax,
                                sigma_K0=30*u.km/u.s,
                                sigma_v=100*u.km/u.s)

    if prior_cache_file is None:
        print("no cache name given")
        prior_cache_file = os.path.join(cache_dir,"simulate_cache.hdf5")

    if os.path.exists(prior_cache_file) is False:
        print("creating cache")
        prior_samples = prior.sample(size=nsamples)#,random_state=rnd)
        print("done sampling")
        prior_samples.write(prior_cache_file, overwrite=True)
        print("created cache")
    else:
        print("reading cache")
        prior_samples = tj.JokerSamples.read(prior_cache_file)
        print("read cache")



    if mpi is True:
        with schwimmbad.MultiPool() as pool:
        # pool = schwimmbad.MultiPool()
            print("Multiprocessing")
            try:
                joker = tj.TheJoker(prior,#random_state=rnd,
                                    pool=pool)
                samples = joker.rejection_sample(data, prior_cache_file,
                                               max_posterior_samples=256)
                print("done sampling")
            except:
                print(name,"Failed")
                return 
    else:
        pool=None
        joker = TheJoker(prior)#,random_state=rnd)
        samples = joker.rejection_sample(data, prior_cache_file,
                                       max_posterior_samples=256)

    samp_per, samp_ecc = samples['P'].value, samples['e'].value
    nsamp = len(samp_per)
    print(len(samples),"Samples")


    fig, ax = plt.subplots(1, 1, figsize=(6,6)) # doctest: +SKIP
    ax.scatter(samples['P'].value, samples['e'].value,
               marker='.', color='k', alpha=0.45) # doctest: +SKIP
    ax.set_xlabel("$P$ [day]")
    ax.set_ylabel("$e$")
    logp = np.log10(samp_per)
    if (max(logp)-min(logp))>1.5:
        ax.set_xscale("log")
    plt.savefig(os.path.join(results_dir,"plots/{0}_P_vs_e.png".format(name)))
    plt.close()
    print("Plotted P vs e")

   # if (nsamp>=1) and (nsamp<50):
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    tmin, tmax = data.t.value[0]-20, 2458450, # to see variation in next few months # data.t.value[-1]+5,
    t_grid = np.linspace(tmin,tmax,5000)
    _ = tj.plot_rv_curves(samples, t_grid, rv_unit=u.km/u.s, ax=ax, data=data,
                   plot_kwargs=dict(color='grey'))
    ax = plt.gca()
    ax.set_xlim(tmin,tmax)
    plt.savefig(os.path.join(results_dir,"plots/{0}_fits.png".format(name)))
    plt.close()
    print("Plotted fits")

#    if (nsamp>=1) and (nsamp<5):
#        for i in range(nsamp):
#            tj.plot_phase_fold(samples[i],data=data)
#            per = samples[i]["P"].to(u.day).value
#            ecc = samples[i]["e"]
#            plt.savefig(os.path.join(results_dir,"plots/{0}_P{1:.2f}_e{2:.2f}_phased.png".format(
#            name,per,ecc)))
#            plt.close()
#        print("plotted phased")
    plt.close("all")
    
    # if nsamp<1000:
    #     at.write(samples,os.path.join(results_dir,"/{0}_rejsamples.csv".format(name)),
    #              delimiter=",",overwrite=True)
    
    if nsamp==1:
        print("1 sample, needs MCMC")

    elif nsamp<256:
        if is_P_unimodal(samples,data):
            print("Needs MCMC")
        else:
            print("insufficient samples! Done with {0}".format(name))
            if pool is not None:
                pool.close()
            return

    else:
        # if pool is not None:
        #     pool.close()
        samples.write(os.path.join(cache_dir,f"joker_samples_{name}.hdf5"),overwrite=True)
        return
    
    # The only things that should continue are the ones that need MCMC
    with prior.model:
        mcmc_init = joker.setup_mcmc(data, samples)

        trace = pm.sample(tune=1000, draws=1000,
                           start=mcmc_init)
                            # Cores/chains are set automatically if not specified
                           # cores=1, chains=2)
    print("MCMC Done")
    if pool is not None:
        pool.close()

    mcmc_samples = tj.JokerSamples.from_inference_data(prior, trace, data)
    mcmc_samples.write(os.path.join(cache_dir,f"joker_samples_mcmc_{name}.hdf5"),overwrite=True)
    
    if to_plot is True:
        mcmc_samples.wrap_K()
        # fig, axes = plt.subplots(nparams, 1, figsize=(6, 3*nparams))
        # for k in range(nparams):
        #     for walker in sampler.chain[..., k]:
        #         axes[k].plot(walker, marker='', drawstyle='steps-mid',
        #                      linewidth=0.5, color='k', alpha=0.2)
        # plt.savefig("plots/{0}_walkers.png".format(name))
        # plt.close()
        # print("plotted walkers")
    
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        _ = tj.plot_rv_curves(mcmc_samples, t_grid=t_grid, data=data,
                   ax=ax, plot_kwargs=dict(color='grey'))
        ax = plt.gca()
        ax.set_xlim(tmin,tmax)
        plt.savefig(os.path.join(results_dir,"plots/{0}_mcmc_fits.png".format(name)))
        plt.close()
        print("plotted mcmc fits")
    
        plt.figure(figsize=(5, 5))
        plt.scatter(mcmc_samples['P'].to(u.day).value,
                    mcmc_samples['e'], linewidth=0, alpha=0.5)
        plt.xlabel("Period [d]")
        plt.ylabel("e")
        plt.savefig(os.path.join(results_dir,"plots/{0}_mcmc_P_vs_e.png".format(name)))
        plt.close()
    
    print("DONE")
    plt.close("all")
    del(prior)
    del(samples)
    try:
        del(mcmc_samples)
    except:
        # They didn't exist
        pass

if __name__=="__main__":


    rv_tab = at.read("test_syn_circ_rvs.csv")

    def get_data(nid,to_delete=None):
        loc = np.where(rv_tab["star"]==nid)[0]

        if to_delete is not None:
            loc = np.delete(loc,to_delete)

        # Set up the RV Data object
        t_raw = rv_tab["JD"][loc]*cds.JD # Need to work the HJD conversion in
        t_day = t_raw.to(u.day)
        t = t_day.value
        rv = rv_tab["rv(km/s)"][loc]*u.km/u.s
        rve = np.ones_like(rv)#*u.km/u.s
        data = tj.RVData(t=t, rv=rv,rv_err=rve)
        return data

    names,uniq_idx = np.unique(rv_tab["star"],return_index=True)

    nrv = np.zeros(len(uniq_idx),int)
    var_std = np.zeros(len(uniq_idx))
    med_rv = np.zeros(len(uniq_idx))

    for i, name in enumerate(names):
        loc = np.where(name==rv_tab["star"])[0]
        nrv[i] = len(loc)
        if len(loc)>3:
            med_rv[i] = np.median(rv_tab["rv(km/s)"][loc])
            std_err = 1.0

            mean, med, std = stats.sigma_clipped_stats(rv_tab["rv(km/s)"][loc])
            var_std[i] = std/std_err
        else:
            med_rv[i] = np.nan


    var = (var_std>3) & (nrv>=3) & np.isfinite(var_std)
    print(len(np.where(var)[0]))
    print(np.nanmax(var_std[var]))
    #var_sort = np.argsort(var_std[var])
    names = names[var]

    for name in names:
        print(name)
        print(datetime.datetime.now())
        data = get_data(name)
        print(os.path.exists(os.path.join(results_dir,"plots/")))
        clown_car(data,name,0.01*u.day,10000*u.day,
                  to_plot=True,nsamples=100_000_000,#int(2e4),
                  prior_cache_file=None,mpi=True)
