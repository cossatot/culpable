
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumtrapz


import attr
from attr.validators import instance_of, optional

from .stats import inverse_transform_sample, pdf_from_samples, Pdf, Cdf



# time-dependent EQ stuff

def RecKDE(data, data_type='samples'):
    # TODO: Make it work for more types of PDFs/input data
    return pdf_from_samples(data, x_min=0, close=False)

 
def S(t, rec_pdf):
    return 1 - rec_pdf.cdf(t)


def hazard(t, rec_pdf):
    #return pdf(t, rec_pdf) / S(t, rec_pdf)
    return rec_pdf(t) / S(t, rec_pdf)


def mean_recurrence_interval(t, rec_pdf):
    return np.trapz(S(t, rec_pdf), t)


def burstiness(rec_ints):
    """Calculates the burstiness parameter as defined by
    Goh and Barabasi, 2008"""
    return ((np.std(rec_ints) - np.mean(rec_ints)) 
            / (np.std(rec_ints) + np.mean(rec_ints)))


def memory(eqs=None, rec_ints=None):
    n = len(rec_ints)
    m = rec_ints.mean()
    v = rec_ints.var()

    return (1 / (n-1)) * np.sum(((rec_ints[i]-m) * (rec_ints[i+1] - m)
                                 for i in range(n-1))) / v



### Earthquake recurrence PDFs

def sample_earthquake_histories(earthquake_list, n_sets, order_check=None):
    """
    Samples earthquake histories based on the timing of individual earthquakes.

    Parameters:
    -----------
    earthquake_list: a list (or tuple) of OffsetMarkers with age information
    n_sets: The number of sample sets generated, i.e. the number of samples per
            event.
    order_check: Any ordering constraints. 
                `None` indicates no constraints.
                `sort` specifies that the sampled events may need to be sorted
                but have no other ordering constrants.
                `trim` specifies that out-of-order samples need to be discarded,
                i.e. if the earthquakes in the list are in stratigraphic order
                but the ages may overlap.


    """

    eq_times = np.array([eq.sample_ages(n_sets) for eq in earthquake_list]).T
    
    if order_check == None:
        eq_times_sort = eq_times

    elif order_check == 'sort':
        eq_times_sort = np.sort(eq_times, axis=1)

    elif order_check == 'trim':
        eq_times_sort = eq_times.copy()
        for i, row in enumerate(eq_times):
            if ~is_monotonic(row):
                while ~is_monotonic(row):
                    row = np.array([eq.sample_ages(1) for eq in eqs.values()])
            eq_times_sort[i,:] = row.T

    return eq_times_sort


def sample_recurrence_intervals(earthquake_histories):

    rec_int_samples = np.diff(earthquake_histories, axis=1)
    
    return rec_int_samples

    
def get_rec_pdf(rec_int_samples):

    if rec_int_samples.shape[0] > 1:
        rec_int_samples = rec_int_samples.ravel()

    rec_int_pdf = RecKDE(rec_int_samples)
    rec_int_pdf.fit()

    return rec_int_pdf

