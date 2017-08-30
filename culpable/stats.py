import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.stats import gaussian_kde

# basic pdf stuff

def normalize_pmf(x, px):

    if x[0] > x[1]:
        denom = np.trapz(px[::-1], x[::-1])
    else:
        denom = np.trapz(px, x)

    if denom != 0.:
        px_norm = px / denom
    else: px_norm = px * 0.

    return x, px_norm



class _Pdf(interp1d):
    def __init__(self, x, px, bounds_error=False, fill_value=0):
        super(_Pdf, self).__init__(x, px, bounds_error=bounds_error,
                                   fill_value=fill_value)
        self.cdf = Cdf(x, px)
        self.icdf = Icdf(x, px)

    def max(self):
        y_max = np.max(self.y)
        x_max = self.x[np.argmax(self.y)]
        return (x_max, y_max)

    def min(self):
        y_min = np.min(self.y)
        x_min = self.x[np.argmin(self.y)]
        return (x_min, y_min)

    def mean(self):
        return pdf_mean(self.x, self.y)

    def score_at_percentile(self, pctile):
        """pctile should be a decimal (between 0. and 1.)"""

        return self.icdf(pctile)

    def median(self):
        return self.score_at_percentile(0.5)



def Pdf(x, px, normalize=True):
    """docstring"""
    if not np.isscalar(x):
        if normalize == True:
            x, px = normalize_pmf(x, px)

        _pdf = _Pdf(x, px, bounds_error=False, fill_value=0.)

    else:
        class DeltaPdf(object):
            def __init__(self, x):
                 self.x = x
            def __call__(val):
                 if val == self.x:
                    return 1.
                 else:
                    return 0.
        _pdf = DeltaPdf(x)

    return _pdf


def Cdf(x, px, normalize=True):
    """docstring"""

    if normalize == True:
        x, px = normalize_pmf(x, px)

    _cdf = interp1d(x, cumtrapz(px, initial=0.) / np.sum(px), 
                    fill_value=1., bounds_error=False)

    return _cdf


def Icdf(x, px, normalize=True):

    if normalize == True:
        x, px = normalize_pmf(x, px)

    _icdf = interp1d(cumtrapz(px, initial=0.) / np.sum(px), x,
                    fill_value=1., bounds_error=False)

    return _icdf

def pdf_mean(x, px):
    return np.trapz(x * px, x)


def trim_pdf(x, px, min=None, max=None):

    if min is not None:
        px = px[x >= min]
        x = x[x >= min] # vals last b/c it gets trimmed
    
    if max is not None:
        px = px[x <= max]
        x = x[x <= max] # vals last b/c it gets trimmed

    x, px = normalize_pmf(x, px)
    return x, px


def pdf_from_samples(samples, n=1000, x_min=None, x_max=None, cut=None, 
                     bw=None, return_arrays=False, close=True):

    _kde = gaussian_kde(samples, bw_method=bw)

    if cut == None:
        bw = _kde.factor
        cut = 3 * bw
    
    if x_min == None:
        x_min = np.min(samples) - cut

    if x_max == None:
        x_max = np.max(samples) + cut

    x = np.linspace(x_min, x_max, n)
    px = _kde.evaluate(x)

    if close == True:
        px[0] = 0.
        px[-1] = 0.

    pdf = Pdf(x, px)

    if return_arrays == True:
        return pdf.x, pdf.y
    else:
        return pdf


def multiply_pdfs(p1, p2, step=None, n_interp=1000):
    x1_min = np.min(p1.x)
    x2_min = np.min(p2.x)
    x1_max = np.max(p1.x)
    x2_max = np.max(p2.x)
    
    x_min = min(x1_min, x2_min)
    x_max = max(x1_max, x2_max)
    
    if step is None:
        x = np.linspace(x_min, x_max, num=n_interp)
    else:
        x = np.arange(x_min, x_max+step, step)
        
    px = p1(x) * p2(x)

    return Pdf(x, px)


def divide_pdfs(p1, p2, step=None, n_interp=1000):
    x1_min = np.min(p1.x)
    x2_min = np.min(p2.x)
    x1_max = np.max(p1.x)
    x2_max = np.max(p2.x)
    
    x_min = min(x1_min, x2_min)
    x_max = max(x1_max, x2_max)
    
    if step is None:
        x = np.linspace(x_min, x_max, num=n_interp)
    else:
        x = np.arange(x_min, x_max+step, step)
        
    
    px = p1(x) / p2(x)

    return Pdf(x, px)




# sampling
def inverse_transform_sample(x, px, n_samps):
    """
    lots o' docs
    """
    if np.isscalar(x) == 1:
        return np.ones(n_samps) * px

    else:
        cdf = Cdf(x, px)

        cdf_interp = interp1d(cdf(x), x, bounds_error=False,
                              fill_value=0.)

        samps = np.random.rand(n_samps)

        return cdf_interp(samps)


def sample_from_bounded_normal(mean, sd, n, sample_min=None, sample_max=None):

    sample = np.random.normal(mean, sd, n)
    sample = trim_distribution(sample, sample_min=sample_min, 
                                       sample_max=sample_max)

    while len(sample) < n:
        next_sample = np.random.normal(mean, sd, n)
        next_sample = trim_distribution(next_sample, sample_min, sample_max)
        sample = np.hstack([sample, next_sample])

    return sample[:n]


def trim_distribution(sample, sample_min=None, sample_max=None):

    if sample_min is not None and sample_max is not None:
        if sample_min >= sample_max:
            raise Exception('min must be less than max!')

    if sample_min is not None:
        sample = sample[sample >= sample_min]

    if sample_max is not None:
        sample = sample[sample <= sample_max]

    return sample


def check_monot_increasing(in_array):
    """Checks to see if array is monotonically increasing, returns bool value
    """
    dx = np.diff(in_array)

    return np.all(dx >= 0)


