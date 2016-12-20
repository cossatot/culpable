import numpy as np

from .stats import Pdf, pdf_from_samples, multiply_pdfs

"""
Scaling relationships and related equations for earthquake magnitude
calculations.
"""


"""
Normalized slip distribution from Biasi and Weldon, 2006
"""
Dn_x = np.array(
    [ 0.        ,  0.03852144,  0.07704287,  0.11556431,  0.15408574,
      0.19260718,  0.23112861,  0.26965005,  0.30817149,  0.34669292,
      0.38521436,  0.42373579,  0.46225723,  0.50077866,  0.5393001 ,
      0.57782153,  0.61634297,  0.65486441,  0.69338584,  0.73190728,
      0.77042871,  0.80895015,  0.84747158,  0.88599302,  0.92451446,
      0.96303589,  1.00155733,  1.04007876,  1.0786002 ,  1.11712163,
      1.15564307,  1.19416451,  1.23268594,  1.27120738,  1.30972881,
      1.34825025,  1.38677168,  1.42529312,  1.46381456,  1.50233599,
      1.54085743,  1.57937886,  1.6179003 ,  1.65642173,  1.69494317,
      1.7334646 ,  1.77198604,  1.81050748,  1.84902891,  1.88755035,
      1.92607178,  1.96459322,  2.00311465,  2.04163609,  2.08015753,
      2.11867896,  2.1572004 ,  2.19572183,  2.23424327,  2.2727647 ,
      2.31128614,  2.34980758,  2.38832901,  2.42685045,  2.46537188,
      2.50389332,  2.54241475,  2.58093619,  2.61945762,  2.65797906,
      2.6965005 ,  2.73502193,  2.77354337,  2.8120648 ,  2.85058624,
      2.88910767,  2.92762911,  2.96615055,  3.00467198,  3.04319342,
      3.08171485,  3.12023629,  3.15875772,  3.19727916,  3.2358006 ,
      3.27432203,  3.31284347,  3.3513649 ,  3.38988634,  3.42840777,
      3.46692921,  3.50545064,  3.54397208,  3.58249352,  3.62101495,
      3.65953639,  3.69805782,  3.73657926,  3.77510069,  3.81362213])

Dn_y = np.array(
    [ 3.56431234e-01,   4.07514412e-01,   4.49469325e-01,
      4.80250978e-01,   4.99600050e-01,   5.08967345e-01,
      5.11056831e-01,   5.09135209e-01,   5.06305810e-01,
      5.04929021e-01,   5.06305202e-01,   5.10647854e-01,
      5.17294850e-01,   5.25056042e-01,   5.32585263e-01,
      5.38688051e-01,   5.42518154e-01,   5.43657945e-01,
      5.42107125e-01,   5.38215229e-01,   5.32589131e-01,
      5.25993774e-01,   5.19250549e-01,   5.13129949e-01,
      5.08236899e-01,   5.04898081e-01,   5.03074847e-01,
      5.02334004e-01,   5.01903866e-01,   5.00822254e-01,
      4.98152675e-01,   4.93216557e-01,   4.85776256e-01,
      4.76112653e-01,   4.64970884e-01,   4.53387277e-01,
      4.42445033e-01,   4.33023117e-01,   4.25598012e-01,
      4.20136711e-01,   4.16092401e-01,   4.12492219e-01,
      4.08093894e-01,   4.01583982e-01,   3.91790171e-01,
      3.77880214e-01,   3.59519131e-01,   3.36956396e-01,
      3.11019404e-01,   2.83002312e-01,   2.54461304e-01,
      2.26954105e-01,   2.01783046e-01,   1.79805426e-01,
      1.61356306e-01,   1.46292387e-01,   1.34126853e-01,
      1.24201482e-01,   1.15842979e-01,   1.08470898e-01,
      1.01650879e-01,   9.51051805e-02,   8.86970782e-02,
      8.24006991e-02,   7.62618151e-02,   7.03540397e-02,
      6.47382510e-02,   5.94357659e-02,   5.44230300e-02,
      4.96471997e-02,   4.50527124e-02,   4.06047119e-02,
      3.62987575e-02,   3.21550847e-02,   2.82040784e-02,
      2.44727150e-02,   2.09786579e-02,   1.77325398e-02,
      1.47440829e-02,   1.20266593e-02,   9.59725861e-03,
      7.47225770e-03,   5.66159378e-03,   4.16411755e-03,
      2.96568107e-03,   2.04006393e-03,   1.35194170e-03,
      8.60866657e-04,   5.25372416e-04,   3.06545806e-04,
      1.70626053e-04,   9.04155999e-05,   4.55329491e-05,
      2.17590136e-05,   9.85449333e-06,   4.22528115e-06,
      1.71367970e-06,   6.56980895e-07,   2.37946616e-07,
      8.13790788e-08])


Dn = Pdf(Dn_x, Dn_y)


"""
Conversion functions
"""

def M_from_D_bw(D):
    """
    Magnitude from displacement using the scaling of Biasi and Weldon (2006).

    Parameters
    ----------
    D : Scalar or vector values for displacement (in meters)


    Returns
    -------
    M : Scalar or vector of calculated magnitude
    """

    return 6.94 + 1.14 * np.log10(D)


def M_from_D_wc(D):
    """
    Magnitude from displacement using the scaling of Wells and Coppersmith '94.

    Parameters
    ----------
    D : Scalar or vector values for displacement (in meters)


    Returns
    -------
    M : Scalar or vector of calculated magnitude
    """

    return 6.93 + 0.82 * np.log(D)


def D_from_M_bw(M):
    """
    Displacement from magnitude using the scaling of Biasi and Weldon (2006).

    Parameters
    ----------
    M : Scalar or vector values for moment magnitude

    Returns
    -------
    D : Scalar or vector of calculated displacement (in meters)
    """

    return 10**((M - 6.94) / 1.14)


def D_from_M_wc(M):
    """
    Displacement from magnitude using the scaling of Wells and Coppersmith '94.

    Parameters
    ----------
    M : Scalar or vector values for moment magnitude

    Returns
    -------
    D : Scalar or vector of calculated displacement (in meters)
    """

    return np.exp((M - 6.93) / 0.82)


def M_from_D(D, ref='bw', a=None, b=None, log='e'):
    """
    Moment magnitude from displacement, using the specified scaling
    (keyword 'ref', or parameters 'a', 'b' and 'log'.

    General relationship is M = a + b * log(D).

    Parameters
    ----------
    D : Scalar or vector values for displacement (in meters)

    ref : string indicating scaling relationship, default 'bw'.
        'bw' is Biasi and Weldon (2006).
        'wc' is Wells and Coppersmith (1994).
        'cus' is custom, using values of 'a', 'b' and 'log'.

    a : Scalar, or vector of same length as D.

    b : Scalar, or vector of same length as D.

    log : String, base for logarithm, default 'e'.
        'e' is natural log.
        '10' is log10.


    Returns
    -------
    M : Scalar or vector of calculated magnitude, with shape of D.
    """

    if ref == 'bw':
        return M_from_D_bw(D)

    elif ref == 'wc':
        return M_from_D_wc(D)

    elif ref == 'cus':

        if log == 'e':
            return a + b * np.log(D)

        elif log in ('10', 10):
            return a + b * np.log10(D)

    else:
        raise NotImplementedError


def D_from_M(M, ref='bw', a=None, b=None, base = 'e'):
    """
    Moment magnitude from displacement, using the specified scaling
    (keyword 'ref', or parameters 'a', 'b' and 'base'.

    General relationship is D = base ** ((M - a) / b)

    Parameters
    ----------
    M : Scalar or vector values for moment magnitude

    ref : string indicating scaling relationship, default 'bw'.
        'bw' is Biasi and Weldon (2006).
        'wc' is Wells and Coppersmith (1994).
        'cus' is custom, using values of 'a', 'b' and 'base'.

    a : Scalar, or vector of same length as M.

    b : Scalar, or vector of same length as M.

    base : String, base for exponent, default 'e'.
        'e' is e.
        '10' is 10.


    Returns
    -------
    D : Scalar or vector of calculated displacement (in meters),
        with shape of M.
    """

    if ref == 'bw':
        return D_from_M_bw(M)

    elif ref == 'wc':
        return D_from_M_wc(M)

    elif ref == 'cus':

        if log == 'e':
            return exp((M - a) / b)

        elif log in ('10', 10):
            return 10. ** ((M - a) / b)

    else:
        raise NotImplementedError


def M_from_L_wc(L):
    """
    Magnitude from displacement using the scaling of Wells and Coppersmith '94.

    Parameters
    ----------
    D : Scalar or vector values for displacement (in meters)


    Returns
    -------
    M : Scalar or vector of calculated magnitude
    """

    return 5.0 + 1.22 * np.log10(L)


def M_from_L(L, ref='wc', unit='km', a=None, a_err=None, b=None, b_err=None,
             log='e', mc=False):
    """
    Moment magnitude from length, using the specified scaling
    (keyword 'ref', or parameters 'a', 'b' and 'log'.

    General relationship is M = a + b * log(D).

    Parameters
    ----------
    D : Scalar or vector values for displacement (in meters)

    ref : string indicating scaling relationship, default 'bw'.
        'bw' is Biasi and Weldon (2006).
        'wc' is Wells and Coppersmith (1994).
        'cus' is custom, using values of 'a', 'b' and 'log'.

    unit : Unit of length measure. Default is 'km'.  'm' also works.

    a : Scalar, or vector of same length as D.
    a_err : Standard error of `a`. Scalar.

    b : Scalar, or vector of same length as D.
    b_err : Standard error of `b`. Scalar.

    log : String, base for logarithm, default 'e'.
        'e' is natural log.
        '10' is log10.


    Returns
    -------
    M : Scalar or vector of calculated magnitude, with shape of L.
    """

    # unit conversion
    if unit == 'm':
        L = L * 1000.

    # calcs
    if ref == 'wc':
        a = 5.08
        a_err = 0.1
        b = 1.16
        b_err = 0.07
        log = 10

    elif ref == 'cus':
        # some checks for parameters?
        pass

    else:
        raise ValueError('{} is not a known reference'.format(ref))

    if mc == True:
        A = a if a_err is None else np.random.normal(a, a_err, len(L))
        B = b if b_err is None else np.random.normal(a, b_err, len(L))
    else:
        A = a
        B = b

    # do calcs
    if log == 'e':
        return A + B * np.log(L)
    elif log in ('10', 10):
        print('log10')
        return A + B * np.log10(L)
    
    else:
        raise NotImplementedError


"""
Estimation functions
"""

def p_D_M(D, M, ref='bw', **kwargs):
    """
    Likelihood of predicted D given M, as defined by Biasi and Weldon (2006).

    Parameters
    ----------
    D : Scalar or array of displacement values (in meters).

    M : Scalar or array of magnitudes.

    ref: Displacement-magnitude scaling reference (string).
        'bw' is Biasi and Weldon (2006).
        'wc' is Wells and Coppersmith (1994).

    Returns
    -------

    p_D_M : Calculated likelihood. If scalar, simply returns the likelihood.
            If not, returns an improper pdf (a `culpable.stats.Pdf`) which
            is an interpolation class. Actual likelihoods are `p_D_M.y`, and
            corresponding magnitudes (i.e. the prior p_M) are `p_D_M.x`.

    """

    D_ave = D_from_M(M, ref=ref, **kwargs)

    D = np.abs(D)

    if np.isscalar(D):
        D_score = D / D_ave 
        p_D_M =  Dn(D_score)
        
    else:
        D_score = np.array([d / D_ave for d in D])

        p_D_M = Dn(D_score)
        p_D_M = np.mean(p_D_M, axis=0)

    if np.isscalar(p_D_M):
        p_D_M = np.float(p_D_M)
    
    else:
        p_D_M = Pdf(M, p_D_M, normalize=True)

    return p_D_M


def make_p_M(p_M_min=6., p_M_max=8., M_step=0.1, n_M=None):
    
    if n_M is not None:
        p_M_x = np.linspace(p_M_min, p_M_max, num=n_M)

    else:
        if M_step is None:
            M_step = 0.1 # in case it's passed as None from another function
        p_M_x = np.arange(p_M_min, p_M_max + M_step, M_step)

    return Pdf(p_M_x, np.ones(len(p_M_x)) * 1 / len(p_M_x))
    

def p_M_D(D, p_M=None, p_M_min=None, p_M_max=None, M_step=None, n_M=None,
          ref='bw', **kwargs):

    """
    Calculates earthquake magnitude given displacement
    """

    if p_M is None:
       p_M = make_p_M(p_M_min, p_M_max, M_step, n_M)

    else:
        #TODO: maybe add some logic for dealing with non `Pdf` priors
        pass

    p_D = np.array([np.trapz(Dn_y, Dn_x * D_from_M(M, ref=ref, **kwargs))
                    for M in p_M.x])

    p_D_M_ = p_D_M(D, p_M.x, ref=ref, **kwargs)

    p_M_D_ = p_M.y * p_D_M_.y / p_D

    p_M_D_ = Pdf(p_M.x, p_M_D_)

    return p_M_D_


def p_M_L(L, p_M=None, p_M_min=None, p_M_max=None, M_step=None, n_M=None,
          ref='wc', mc=True, **kwargs):

    if p_M is None:
        p_M = make_p_M(p_M_min, p_M_max, M_step, n_M)


    p_M_L_samples = M_from_L(L, ref=ref, mc=mc, **kwargs)

    p_M_L_ = pdf_from_samples(p_M_L_samples, x_min=p_M.x.min(),
                              x_max=p_M.x.max())

    p_M_L_ = multiply_pdfs(p_M, p_M_L_)

    return p_M_L_


def p_M_DL(D, L, p_M=None, p_M_min=None, p_M_max=None, M_step=None, n_M=None,
           ref='wc', L_mc=True, **kwargs):
    
    if p_M is None:
        p_M = make_p_M(p_M_min, p_M_max, M_step, n_M)

    p_M_D_ = p_M_D(D, p_M, ref=ref, **kwargs)

    p_M_L_samples = M_from_L(L, ref=ref, mc=L_mc, **kwargs)

    p_M_L_ = pdf_from_samples(p_M_L_samples, x_min=p_M.x.min(), 
                              x_max=p_M.x.max())

    return multiply_pdfs(p_M_L_, p_M_D_)
