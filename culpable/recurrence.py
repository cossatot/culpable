
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumtrapz
from statsmodels.nonparametric.kde import KDEUnivariate


import attr
from attr.validators import instance_of, optional

from .stats import inverse_transform_sample

@attr.s
class RecKDE(object):
    # consider generalizing this w/ kde in stats module,
    # or using scipy gaussian_kde to remove statsmodels dependency

    data = attr.ib(default=attr.Factory(np.array), convert=np.array,
                   validator=instance_of(np.array))
    
    def fit(self, x_min = 0., **kwargs):

        kde = KDEUnivariate(self.data)
        kde.fit(**kwargs)
        
        xx = kde.support[kde.support > x_min]
        px = kde.density[kde.support > x_min]
        px /= np.trapz(px, xx)
        
        self.x = xx
        self.px = px
        
    def Pdf(self, t=None):
        if t is None:
            t = self.x
        return pdf(t, self)
    
    def Cdf(self, t=None):
        if t is None:
            t = self.x
        return cdf(t, self)
    
    
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


### stuff to bring in from old eq_recurrence.py
def sample_eq_pdf(row, n_samps):
    #q_vals, eq_probs = make_eq_pdfs(row['age_5'], row['age_mean'], 
    #                                 row['age_95'])
    #
    #return inverse_transform_sample(eq_vals, eq_probs, n_samps)

    #TODO: This needs to be made more general, using the OffsetMarker class

    raise NotImplementedError


def make_eq_time_series(eq_list, n_samples):
    # need to implement sample_eq_pdf first
    return np.array([sample_eq_pdfs(eq, n_samples) for eq in eq_list]).T


def calculate_recurrence_intervals(eq_time_series):
    return np.diff(np.sort(eq_time_series, axis=1), axis=1)
