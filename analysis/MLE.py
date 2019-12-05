#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import numpy as np
import warnings
import os


def MaximumLikelihood(vals, export_asymp_ci = False, fix = 0, metod = 'L-BFGS-B'):
    from scipy.interpolate import interp1d
    from scipy.optimize import minimize
    from scipy import special
    from scipy.stats import poisson,norm
    from scipy.special import j_roots
    from scipy.special import beta as beta_fun
    import numpy as np
    if len(vals) == 0:
        return np.array([np.nan, np.nan, np.nan])
    def dBP(at, alpha, bet, lam):
        at.shape = (len(at), 1)
        np.repeat(at, 50, axis = 1)
        def fun(at, m):
            if(max(m) < 1e6):
                return(poisson.pmf(at,m))
            else:
                return(norm.pdf(at,loc=m,scale=sqrt(m)))

        x,w = j_roots(50,alpha = bet - 1, beta = alpha - 1)
        gs = np.sum(w*fun(at, m = lam*(1+x)/2), axis=1)
        prob = 1/beta_fun(alpha, bet)*2**(-alpha-bet+1)*gs
        return(prob)
    def LogLikelihood(x, vals):
        kon = x[0]
        koff = x[1]
        ksyn = x[2]
        return(-np.sum(np.log( dBP(vals,kon,koff,ksyn) + 1e-10) ) )
    x0 = MomentInference(vals)
    if np.isnan(x0).any() or any(x0 < 0):
        x0 = np.array([10,10,10])
    bnds = ((1e-3,1e3),(1e-3,1e3), (1, 1e4))
    vals_ = np.copy(vals) # Otherwise the structure is violated.
    try:
        ll = minimize(LogLikelihood, x0, args = (vals_), method=metod, bounds=bnds)
    except:
        return np.array([np.nan,np.nan,np.nan])
    #se = ll.hess_inv.todense().diagonal()
    estim = ll.x
    return estim

# moment-based inference
def MomentInference(vals, export_moments=False):
    # code from Anton Larsson's R implementation
    from scipy import stats # needs imports inside function when run in ipyparallel
    import numpy as np
    m1 = float(np.mean(vals))
    m2 = float(sum(vals*(vals - 1))/len(vals))
    m3 = float(sum(vals*(vals - 1)*(vals - 2))/len(vals))

    # sanity check on input (e.g. need at least on expression level)
    if sum(vals) == 0: return np.nan
    if m1 == 0: return np.nan
    if m2 == 0: return np.nan

    r1=m1
    r2=m2/m1
    r3=m3/m2

    if (r1*r2-2*r1*r3 + r2*r3) == 0: return np.nan
    if ((r1*r2 - 2*r1*r3 + r2*r3)*(r1-2*r2+r3)) == 0: return np.nan
    if (r1 - 2*r2 + r3) == 0: return np.nan

    lambda_est = (2*r1*(r3-r2))/(r1*r2-2*r1*r3 + r2*r3)
    mu_est = (2*(r3-r2)*(r1-r3)*(r2-r1))/((r1*r2 - 2*r1*r3 + r2*r3)*(r1-2*r2+r3))
    v_est = (2*r1*r3 - r1*r2 - r2*r3)/(r1 - 2*r2 + r3)

    if export_moments:
        return np.array([lambda_est, mu_est, v_est, r1, r2, r3])

    return np.array([lambda_est, mu_est, v_est])



def generate_data(kon,koff,kr,n_cells):
    x=np.empty([n_cells, 1])
    for i in range(0,n_cells):
        p=np.random.beta(kon, koff, size=None)
        x[i]=np.random.poisson(kr*p, size=None)
    return(x)

	
n_estimates=1000
##small k_on and k_off
est_1=x=np.empty([n_estimates, 3])
for j in range(0,n_estimates):
    x=generate_data(0.1,0.1,100,100)
    est_1[j]=MaximumLikelihood(x)
	
	
##large k_on and k_off
est_2=x=np.empty([n_estimates, 3])
for j in range(0,n_estimates):
    x=generate_data(10,10,100,100)
    est_2[j]=MaximumLikelihood(x)
	
	
##small k_on and large k_off
est_3=x=np.empty([n_estimates, 3])
for j in range(0,n_estimates):
    x=generate_data(1,10,100,100)
    est_3[j]=MaximumLikelihood(x)
	
	
##Large k_on and small k_off
est_4=x=np.empty([n_estimates, 3])
for j in range(0,n_estimates):
    x=generate_data(10,1,100,100)
    est_4[j]=MaximumLikelihood(x)