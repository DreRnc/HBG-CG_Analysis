import numpy as np

def f2phi(f, theta, d, alpha):

    '''
    Computes tomography and its derivative.

    Parameters
    ----------
    f
    theta
    d
    alpha


    Returns
    -------
    phi
    phip

    '''

    phi = f(theta + alpha * d)
    phip = d * f(theta + alpha * d)

    return phi, phip


def AWLS(J, theta, d, m1, m2, alpha_s, tau, sfgrd, min_alpha, delta, eps, MaxFeval):
    '''
    Inexact line search. Computes the step-size satisfying the AW conditions.

    Parameters
    ----------

    J :
    theta (np.array) : 
    d (np.array) :
    m1 (Float) : 
    m2 (Float) : 
    alpha_s (Float) : 
    tau (Float) : 
    sfgrd (Float) : 
    delta (Float) :
    eps (Float) : 
    MaxFeval (int):


    Returns
    -------
    alpha (Float) : Step size
    '''
    feval = 1
    [ phi0 , phip0 ] = f2phi( J, theta, d, 0)

    while feval <= MaxFeval:
        [ phia , phips ] = f2phi( J, theta, d, alpha_s )
        
        if ( phia <= phi0 + m1 * alpha_s * phip0 ) and ( np.abs( phips ) <= - m2 * phip0 ):
           alpha = alpha_s
           return alpha
           
           # Armijo + strong Wolfe satisfied, we are done
           
        if phips >= 0:  # derivative is positive
           break
        
        alpha_s = alpha_s / tau
        
    alpha_m = 0;
    alpha = alpha_s;
    phipm = phip0;
    
    while ( feval <= MaxFeval ) and ( ( alpha_s - alpha_m ) ) > delta and ( phips > eps ):
       # compute the new value by safeguarded quadratic interpolation
        
       alpha = ( alpha_m * phips - alpha_s * phipm ) / ( phips - phipm );
       alpha = np.max( np.array([alpha_m + ( alpha_s - alpha_m ) * sfgrd, \
                  np.min( np.array([alpha_s - ( alpha_s - alpha_m ) * sfgrd, alpha]) ) ]) )
       
       # compute tomography
        
       [ phia , phip ] = f2phi( J, theta, d, alpha )
       
       if ( phia <= phi0 + m1 * alpha * phip0 ) and ( np.abs( phip ) <= - m2 * phip0 ):
          break #Armijo + strong Wolfe satisfied, we are done

       # restrict the interval based on sign of the derivative in a
       if phip < 0:
          alpha_m= alpha
          phipm = phip
       else:
          alpha_s = alpha
          if alpha_s <= min_alpha:
             break
          phips = phip

    return alpha