
import numpy as np 
import xarray as xr 
import pyhomogeneity as hg
import statsmodels.api as sm

# GEV parameter calculation for xarray
def xr_homotest(xarray,dim = 'time', test = 'pettitt'):#,alpha=0.05,sim = 2000):    
    ''' This function homogenity test results for xarray object in one dimension.
    Parameters:
    xarray: Xarray with the variable 
    dim: dimension to apply test function
    test: name of the test (pettitt by default)
    
    Return:
        xarray dataset with all paramters and return value 
        h: boolean for hipotesis test 
        cp: date of change point 
        p: p-value
        U: test parameter
        m1: mean before change point
        m2: mean after change point
    '''
    ds = xr.Dataset()
    if test == 'pettitt':
        f = hg.pettitt_test
    elif test == 'snht_test':
        f = hg.snht_test
    elif test == 'buishand_u_test':
        f = hg.buishand_u_test
    
    def newf(f,x):
        a,b,c,d,g = f(x)#,kwargs = {'alpha': alpha,'sim':sim})
        g1 = g[0]
        g2 = g[1]
        return (a,b,c,d,g1,g2)
    func = lambda x: newf(f,x)
    h, cp, p, U, m1,m2 = xr.apply_ufunc(func,xarray,input_core_dims = [[dim]],
                        output_core_dims = [[] for _ in range(6)],
                        #kwargs={'alpha': alpha,'sim':sim},
                        #kwargs={'f':hg.pettitt_test},
                        # exclude_dims=set(("time",)),                        
                        vectorize = True,
                        )     
    ds['h'] = h
    ds['cp'] = cp
    ds['p'] = p
    ds['U'] = U
    ds['m1'] = m1
    ds['m2'] = m2
    return ds