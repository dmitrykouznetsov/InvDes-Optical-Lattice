# The operators functions are based on the examples found in https://github.com/fancompute/ceviche

import autograd.numpy as npa
from skimage.draw import disk
from autograd.scipy.signal import convolve as conv
import numpy as np


def operator_proj(rho, eta=0.5, beta=100, N=1):
    ''' Density projection driving rho towards "binarized" design
    eta: Center of projection between 0 and 1
    beta: Strength of the projection
    N: Number of times to apply the projection
    '''
    for i in range(N):
        rho = npa.divide(npa.tanh(beta * eta) + npa.tanh(beta * (rho - eta)),
                         npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta)))

    return rho


def _create_blur_kernel(radius):
    ''' Helper function for smoothing features (for conv kernel) '''
    rr, cc = disk((radius, radius), radius+1)
    kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float)
    kernel[rr, cc] = 1
    return  kernel / kernel.sum()


def operator_blur(rho, radius=2, N=1):
    ''' Blur operator via 2d convolution
    radius: Radius of convolution kernel
    N: Number of times to apply the filter
    '''
    kernel = _create_blur_kernel(radius)

    for i in range(N):
        # For whatever reason HIPS autograd doesn't support 'same' mode, so we need to manually crop the output
        rho = conv(rho, kernel, mode='full')[radius:-radius,radius:-radius]

    return rho


def mask_combine_rho(rho, bg_rho, design_region):
    ''' Utility function for combining the design region rho and the background rho '''
    return rho*design_region + bg_rho*(design_region==0).astype(np.float)


def epsr_parameterization(rho, bg_rho, design_region, params):
    ''' Defines the parameterization steps for constructing rho '''
    # Combine rho and bg_rho; Note: this is so the subsequent blur sees the waveguides
    rho = mask_combine_rho(rho, bg_rho, design_region)

    rho = operator_blur(rho, radius=params.blur_radius, N=params.blur_n)
    rho = operator_proj(rho, beta=params.beta, eta=params.eta, N=params.proj_n)

    # Final masking undoes the blurring of the waveguides
    rho = mask_combine_rho(rho, bg_rho, design_region)

    return params.epsr_min + (params.epsr_max-params.epsr_min) * rho
