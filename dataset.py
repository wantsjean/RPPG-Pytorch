import numpy as np
from scipy.ndimage import gaussian_filter1d
a = np.array([30,200,77,98])
a = 1/a
filted = gaussian_filter1d(a,sigma=1)
print(filted)