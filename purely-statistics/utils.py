import numpy as np
import numpy.testing as npt
from scipy.stats import entropy


def arrays_to_pmf(ndarray1, ndarray2, bins):
    #use the bins of the larger array
    min_array = ndarray1.min(
    ) if ndarray2.min() > ndarray1.min() else ndarray2.min()
    max_array = ndarray1.max(
    ) if ndarray2.max() < ndarray1.max() else ndarray2.max()
    hist_range = (min_array, max_array)
    ahist, bhist = (np.histogram(ndarray1, bins=bins, range=hist_range)[0],
                    np.histogram(ndarray2, bins=bins, range=hist_range)[0])

    # Here we need to make a Laplace correction, where we add 1 count to each bin,
    # and renormalize. This way, the pdf has mass everywhere, this is importance since,
    # KL divergence does not accept zeros
    ahist, bhist = ahist + 1, bhist + 1
    a_pdf, b_pdf = ahist / np.sum(ahist), bhist / np.sum(bhist)
    npt.assert_almost_equal(np.sum(a_pdf), 1, decimal=5)
    npt.assert_almost_equal(np.sum(b_pdf), 1, decimal=5)
    return a_pdf, b_pdf


def KL(cls, ndarray1, ndarray2, bins=50):
    p, q = arrays_to_pmf(ndarray1, ndarray2, bins)
    return entropy(p, qk=q)


def KS(cls, ndarray1, ndarray2):
    """
    Kolmogorov-Smirnov statistic

    Notes
    -----
    KS hypothesis: Distributions generating these 2 arrays are the same
    If the K-S statistic is small or the p-value is high, then we cannot 
    reject the hypothesis that the distributions of the two samples are the same.

    Generally, if the p-value is bellow 1% we can reject the null hypothesis.

    The test uses the two-sided asymptotic Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    ndarray1: np.ndarray
        concatenated metrics from one rollout
    ndarray2: np.ndarray
        concatenated metrics from a second rollout
    """

    statistics = ks_2samp(ndarray1, ndarray2)
    return statistics.statistic, statistics.pvalue