import numpy as np
from scipy.stats import sigmaclip


def Sigmaclip(array, low=4., high=4, axis=None):
    '''(ndarray, int, int, int) -> ndarray

    Iterative sigma-clipping of along the given axis.
	   
    The output array contains only those elements of the input array `c`
    that satisfy the conditions ::
	   
    mean(c) - std(c)*low < c < mean(c) + std(c)*high
	
    Parameters
    ----------
    a : array_like
    data array
    low : float
    lower bound factor of sigma clipping
    high : float
    upper bound factor of sigma clipping
    
    Returns
    -------
    c : array
    sigma clipped mean along axis
    '''
    c = np.asarray(array)
    if axis is None or c.ndim == 1:
        #from scipy.stats import sigmaclip
        out = sigmaclip(c)[0]
        return out.mean(), out.std()
    #create masked array
    c_mask = np.ma.masked_array(c, np.isnan(c))
    delta = 1
    while delta:
           c_std = c_mask.std(axis=axis)
           c_mean = c_mask.mean(axis=axis)
           size = c_mask.mask.sum()
           critlower = c_mean - c_std*low
           critupper = c_mean + c_std*high
           indexer = [slice(None)] * c.ndim
           for i in xrange(c.shape[axis]):
               indexer[axis] = slice(i,i+1)
               c_mask[indexer].mask = np.logical_and(
                   c_mask[indexer].squeeze() > critlower, 
                   c_mask[indexer].squeeze() < critupper) == False
           delta = size - c_mask.mask.sum()
    return c_mask.mean(axis).data, c_mask.std(axis).data

