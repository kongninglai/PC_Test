import numpy as np
import typing

def xlogy(x1, x2, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None, extobj=None) -> typing.Any:
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    # Use numpy's where function to ensure xlogy(0, y) = 0 even if y is 0
    result = np.where(x1 == 0, 0, x1 * np.log(x2))
    
    if out is not None:
        out[...] = result
    
    return result if out is None else out

def prod(iters, start = 1):
    result = start
    try:
        for iter in iters:
            result = result * iter 
    except:
        pass 

    return result