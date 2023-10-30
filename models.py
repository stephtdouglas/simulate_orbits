def circ_function(Porb,Pcirc,alpha=0.35,beta=0.14,gamma=1.0):
    """
    Compute the eccentricity distribution for the input
    Porb values. Alpha, beta, and gamma are taken from 
    Meibom & Mathieu (2005). 
    
    Inputs:
        Porb: array-like (Quantity optional)
        Pcirc: float (Quantity optional)
        alpha, beta, gamma: floats (optional)
        
    Outputs:
        eccentricity: array
    """
    
    eccentricities = np.zeros_like(Porb)
    
    # If Porb <= Pcirc, then e=0, so we don't need to change anything
    
    # If Porb > Pcirc, compute the circularization function
    
    gtr_pc = Porb > Pcirc
    
    cf_part = np.e**(beta * (Pcirc - Porb[gtr_pc]))
    
    eccentricities[gtr_pc] = alpha * (1 - cf_part)**gamma
    
    return eccentricities

# Following Zanazzi+2022, derive an envelope function
def find_env(per,ecc):
    nbins = 20
    pbins = np.logspace(0,3,nbins)
    eenv = np.zeros(nbins)
    for i in range(nbins-1):
        loc = np.where((per>=pbins[i]) & (per<pbins[i+1]))[0]
        if len(loc)==0:
            print(pbins[i])
            eenv[i] = eenv[i-1]
        else:
            eenv[i] = max(ecc[loc])
    return pbins, eenv

