import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumtrapz
from scipy.stats import gaussian_kde

# basic pdf stuff
def make_pdf(vals, probs, n_interp=1000):
    """
    Takes a sequence of x (values) and p(x) (probs) and makes a PDF
    """
    
    
    val_min = np.min(vals)
    val_max = np.max(vals)
    
    # if the PDF is just a point (no uncertainty)
    if val_min == val_max: 
        return val_min, 1.
    
    # if not...
    else:
        pdf_range = np.linspace(val_min, val_max, n_interp)

        pmf = interp1d(vals, probs)
        pmf_samples = pmf(pdf_range)
        pdf_probs = pmf_samples / np.sum(pmf_samples) # normalize

    return pdf_range, pdf_probs


def make_cdf(pdf_range, pdf_probs):
    """ Makes a CDF from a PDF """
    return (pdf_range, np.cumsum(pdf_probs))


def pdf_mean(pdf_vals, pdf_probs):
    return trapz(pdf_vals * pdf_probs, pdf_vals)


def pdf_from_samples(samples, n=100, x_min=None, x_max=None, cut=None):
    _kde = gaussian_kde(samples)

    if cut == None:
        bw = _kde.factor
        cut = 3 * bw
    
    if x_min == None:
        x_min = np.min(samples) - cut

    if x_max == None:
        x_max = np.max(samples) + cut

    support = np.linspace(x_min, x_max, n)
    px = _kde.evaluate(support)
    px /= np.trapz(px, support)

    return support, px


# sampling
def inverse_transform_sample(vals, probs, n_samps, n_interp=1000):
    """
    
    """
    pdf_range, pdf_probs = make_pdf(vals, probs, n_interp)
    cdf_range, cdf_probs = make_cdf(pdf_range, pdf_probs)

    if len(cdf_probs) == 1:
        return np.ones(n_samps) * pdf_range
    
    else:
        cdf_interp = interp1d(cdf_probs, cdf_range, bounds_error=False,
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


