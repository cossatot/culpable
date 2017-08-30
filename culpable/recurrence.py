
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumtrapz
from statsmodels.nonparametric.kde import KDEUnivariate


import attr
from attr.validators import instance_of, optional

from .stats import inverse_transform_sample, pdf_from_samples, Pdf, Cdf



# time-dependent EQ stuff
@attr.s
class RecKDE(object):
    # consider generalizing this w/ kde in stats module,
    # or using scipy gaussian_kde to remove statsmodels dependency

    data = attr.ib(default=attr.Factory(np.array), #convert=np.array,
                   #validator=instance_of(np.array)
                   )

    def __attrs_post_init__(self):
        #self.x, self.y = pdf_from_samples(self.data, x_min=0, close=False,
        #                                  return_arrays=True)
        self = pdf_from_samples(self.data, x_min=0, close=False)
        self.px = self.y

    
    
def pdf(t, rec_pdf):
    pdf_ = interp1d(rec_pdf.x, rec_pdf.px, kind='linear',
                    bounds_error=False, fill_value=0.)
    return pdf_(t)


def cdf(t, rec_pdf):
    cdf_ = interp1d(rec_pdf.x, 
                    cumtrapz(rec_pdf.px, rec_pdf.x, initial=0.),
                    kind='linear', bounds_error=False, fill_value=1.)
    return cdf_(t)


def S(t, rec_pdf):
    return 1 - cdf(t, rec_pdf)


def hazard(t, rec_pdf):
    return pdf(t, rec_pdf) / S(t, rec_pdf)


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

