# Copyright 2018 Google Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
The function for computing projection weightings.

See:
https://arxiv.org/abs/1806.05759
for full details.

"""

import numpy as np
import cca_core

def compute_pwcca(acts1, acts2, epsilon=0.):
    """ Computes projection weighting for weighting CCA coefficients 
    
    Args:
         acts1: 2d numpy array, shaped (neurons, num_datapoints)
	 acts2: 2d numpy array, shaped (neurons, num_datapoints)

    Returns:
	 Original cca coefficient mean and weighted mean

    """
    sresults = cca_core.get_cca_similarity(acts1, acts2, epsilon=epsilon, 
					   compute_dirns=False, compute_coefs=True, verbose=False)
    if np.sum(sresults["x_idxs"]) <= np.sum(sresults["y_idxs"]):
        dirns = np.dot(sresults["coef_x"], 
                    (acts1[sresults["x_idxs"]] - \
                     sresults["neuron_means1"][sresults["x_idxs"]])) + sresults["neuron_means1"][sresults["x_idxs"]]
        coefs = sresults["cca_coef1"]
        acts = acts1
        idxs = sresults["x_idxs"]
    else:
        dirns = np.dot(sresults["coef_y"], 
                    (acts1[sresults["y_idxs"]] - \
                     sresults["neuron_means2"][sresults["y_idxs"]])) + sresults["neuron_means2"][sresults["y_idxs"]]
        coefs = sresults["cca_coef2"]
        acts = acts2
        idxs = sresults["y_idxs"]
    P, _ = np.linalg.qr(dirns.T)
    weights = np.sum(np.abs(np.dot(P.T, acts[idxs].T)), axis=1)
    weights = weights/np.sum(weights)
    
    return np.sum(weights*coefs), weights, coefs 
