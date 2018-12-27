
# Plotting tools for statistics

import matplotlib.pyplot as plt
import math
import numpy as np

def normal_plot(mu,sigma,x=None):
    """
    One-dimensional normal distribution plot (scalar mean, and var)

    x - accepts an object of type np.linspace
    mu - mean of the normal distribution (scalar)
    sigma - variance of the normal distribution (scalar)
    
    Shows a matplotlib plot on call
    """

    if (x == None):
        x = np.linspace(mu-3*sigma,mu+3*sigma,1000)

    # Variance
    var = sigma**2

    # Normalisation constant
    c = np.sqrt(2 * math.pi * var)

    # p(x) mapping, note: could be simplified
    p = np.exp(-(x-mu)**2/ (2*var) ) / c

    plt.plot(x,p)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.show()

