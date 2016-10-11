import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz, cumtrapz


def make_pdf(vals, probs, n_interp=1000):
    """
    Takes a sequence of 
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


def pdf_mean(pdf_vals, pdf_probs):
    return trapz(pdf_vals * pdf_probs, pdf_vals)

