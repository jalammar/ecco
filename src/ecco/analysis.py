from .svcca_lib import cca_core, pwcca as pwcca_lib, cka_lib

import numpy as np


def cca(acts1, acts2):
    """
    Calculate a similarity score for two activation matrices using Canonical Correlation Analysis (CCA). Returns the
    average of all the correlation coefficients.
    Args:
        acts1: Activations matrix #1. 2D numPy array. Dimensions: (neurons, token position)
        acts2: Activations matrix #2. 2D numPy array. Dimensions: (neurons, token position)

    Returns:
        score: Float between 0 and 1, where 0 means not correlated, 1 means the two activation matrices are linear transformations of each other.
    """
    result = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-10, verbose=False)
    return result['mean'][0]


# More details at https://github.com/google/svcca/blob/master/tutorials/001_Introduction.ipynb
def svcca(acts1, acts2, dims: int = 20):
    """
    Calculate a similarity score for two activation matrices using Singular Value Canonical Correlation Analysis
    (SVCCA). A meaningful score requires setting an appropriate value for 'dims', see SVCCA tutorial for how to do
    that.
    Args:
        acts1: Activations matrix #1. 2D numPy array. Dimensions: (neurons, token position)
        acts2: Activations matrix #2. 2D numPy array. Dimensions: (neurons, token position)
        dims: The number of dimensions to consider for SVCCA calculation. See the SVCCA tutorial to see how to
                determine this in a way

    Returns:
        score: between 0 and 1, where 0 means not correlated, 1 means the two activation matrices are linear
        transformations of each other.
    """
    # Center activations by subtracting the mean
    centered_acts_1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    centered_acts_2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    _, s1, v1 = np.linalg.svd(centered_acts_1, full_matrices=False)
    _, s2, v2 = np.linalg.svd(centered_acts_2, full_matrices=False)

    # Reconstruct by multiplying s and v but only for the top <dims> number of dimensions
    sv_acts1 = np.dot(s1[:dims] * np.eye(dims), v1[:dims])
    sv_acts2 = np.dot(s2[:dims] * np.eye(dims), v2[:dims])

    # NOW do cca to get SVCCA values
    results = cca_core.get_cca_similarity(sv_acts1, sv_acts2, epsilon=1e-10, verbose=False)

    return np.mean(results["cca_coef1"])


def pwcca(acts1, acts2, epsilon=1e-10):
    """
    Calculate a similarity score for two activation matrices using Projection Weighted Canonical Correlation Analysis.
    It's more convenient as it does not require setting a specific number of dims like SVCCA.
    Args:
        acts1: Activations matrix #1. 2D numPy array. Dimensions: (neurons, token position)
        acts2: Activations matrix #2. 2D numPy array. Dimensions: (neurons, token position)

    Returns:
        score: between 0 and 1, where 0 means not correlated, 1 means the two activation matrices are
        linear transformations of each other.
    """
    results = pwcca_lib.compute_pwcca(acts1, acts2, epsilon=epsilon)
    return results[0]

def cka(acts1, acts2):
    """
    Calculates a similarity score for two activation matrices using center kernel alignment (CKA). CKA
    has the benefit of not requiring the number of tokens to be larger than the number of neurons.

    Args:
        acts1: Activations matrix #1. 2D numPy array. Dimensions: (neurons, token position)
        acts2: Activations matrix #2. 2D numPy array. Dimensions: (neurons, token position)

    Returns:
        score: between 0 and 1, where 0 means not correlated, 1 means the two activation matrices are
        linear transformations of each other.
    """

    return cka_lib.feature_space_linear_cka(acts1.T, acts2.T)