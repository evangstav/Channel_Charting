import scipy.io
import numpy as np

def get_data_from_mat(filename):
    """
    A function to get the data from the mat files in a tuple 
    """
    features = ['Scaling', 'SamplingRate', 'x']
    data_m = scipy.io.loadmat(filename)
    scale, sampling, z = (data_m[feature].flatten() for feature in features)
    real, imag = (scipy.signal.resample(np.real(z), len(np.real(z))//111),
                  scipy.signal.resample(np.imag(z), len(np.imag(z))//111))
    return real.tolist(), imag.tolist(), int(scale[0]), float(sampling[0])
