from functools import partial
from time import time
import torch
from typing import Any, Dict
from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    DeepLift,
    DeepLiftShap,
    GuidedBackprop,
    GuidedGradCam,
    Deconvolution,
    LRP,
    Lime,
    LimeBase,
    KernelShap,
    GradientShap,
    Occlusion,
)
import numpy as np
import numpy.linalg as la
from torch.nn import functional as F
import transformers

IG_N_STEPS=50

ATTR_NAME_ALIASES = {
    'ig': 'integrated_gradients',
    'saliency': 'gradient',
    'dl': 'deep_lift',
    'dls': 'deep_lift_shap',
    'gb': 'guided_backprop',
    'gg': 'guided_gradcam',
    'deconv': 'deconvolution',
    'lrp': 'layer_relevance_propagation',
    'lime': 'lime',
    'limebase': 'limebase',
    'shap': 'shap',
    'gshap': 'gshap',
    'occlusion': 'occlusion'
}

ATTR_NAME_TO_CLASS = { # TODO: Add more Captum Primary attributions with needed computed arguments
    'integrated_gradients': IntegratedGradients,
    'gradient': Saliency,
    'grad_x_input': InputXGradient,
    'deep_lift': DeepLift,
    'deep_lift_shap': DeepLiftShap,
    'guided_backprop': GuidedBackprop,
    'guided_gradcam': GuidedGradCam,
    'deconvolution': Deconvolution,
    'layer_relevance_propagation': LRP,
    'lime': Lime,
    'limebase': LimeBase,
    'shap': KernelShap,
    'gshap': GradientShap,
    'occlusion': Occlusion
}


def compute_primary_attributions_scores(attr_method : str, model: transformers.PreTrainedModel,
                                        forward_kwargs: Dict[str, Any], prediction_id: torch.Tensor,
                                        supertoken_range: [],
                                        aggregation: str = "L2") -> torch.Tensor:
    """
    Computes the primary attributions with respect to the specified `prediction_id`.

    Args:
        attr_method: Name of the primary attribution method to compute
        model: HuggingFace Transformers Pytorch language model.
        forward_kwargs: contains all the inputs that are passed to `model` in the forward pass
        prediction_id: Target Id. The Integrated Gradients will be computed with respect to it.
        aggregation: Aggregation/normalzation method to perform to the Integrated Gradients attributions.
         Currently only "L2" is implemented

    Returns: a tensor of the normalized attributions with shape (input sequence size,)

    """

    def model_forward(input_: torch.Tensor, decoder_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
            -> torch.Tensor:
        if decoder_ is not None:
            output = model(inputs_embeds=input_, decoder_inputs_embeds=decoder_, **extra_forward_args)
        else:
            output = model(inputs_embeds=input_, **extra_forward_args)
        return F.softmax(output.logits[:, -1, :], dim=-1)

    def normalize_attributes(attributes: torch.Tensor) -> torch.Tensor:
        # attributes has shape (batch, sequence size, embedding dim)
        attributes = attributes.squeeze(0)

        if aggregation == "L2":  # norm calculates a scalar value (L2 Norm)
            norm = torch.norm(attributes, dim=1)
            attributes = norm / torch.sum(norm)  # Normalize the values so they add up to 1
        else:
            raise NotImplemented

        return attributes

    extra_forward_args = {k: v for k, v in forward_kwargs.items() if
                          k not in ['inputs_embeds', 'decoder_inputs_embeds']}
    input_ = forward_kwargs.get('inputs_embeds')
    decoder_ = forward_kwargs.get('decoder_inputs_embeds')

    # for dec-only models
    if decoder_ is None:
        forward_func = partial(model_forward, decoder_=decoder_, model=model, extra_forward_args=extra_forward_args)
        inputs = input_
    # for enc-dec models
    else:
        forward_func = partial(model_forward, model=model, extra_forward_args=extra_forward_args)
        inputs = tuple([input_, decoder_])

    attr_method_class = ATTR_NAME_TO_CLASS.get(ATTR_NAME_ALIASES.get(attr_method, attr_method), None)
    if attr_method_class is None:
        raise NotImplementedError(
            f"No implementation found for primary attribution method '{attr_method}'. "
            f"Please choose one of the methods: {list(ATTR_NAME_TO_CLASS.keys())}"
        )

    # ig = attr_method_class(forward_func=forward_func, multiply_by_inputs=True) # for [saliency, ig]
    ig = attr_method_class(forward_func=forward_func) # for [lime, shap]
    
    # print("inputs shape is", inputs.shape)
    # attributions = ig.attribute(inputs, target=prediction_id, n_steps=IG_N_STEPS) # for [ig]
    # attributions = ig.attribute(inputs, target=prediction_id) # for [saliency, lime, shap]
    
    # feature_mask should be of size torch.Size([1, 216, 768]), with all the same number in each row
    # like this: [[[0, 0, 0, ..., 0, 0, 0], [1, 1, 1, ..., 1, 1, 1], ..., [215, 215, 215, ..., 215, 215, 215]]]
    feature_mask = torch.zeros(inputs.shape, dtype=torch.long)
    for i in range(inputs.shape[1]):
        feature_mask[0][i] = i
    feature_mask = feature_mask.to(inputs.device)
    
    # feature_mask should be of size torch.Size([1, 216, 768]), with all the same numbe for each citation range
    # feature_mask = torch.zeros(inputs.shape, dtype=torch.long)
    # num_citations = len(supertoken_range)
    # j = 0
    # for i in range(inputs.shape[1]):
    #     # a trick here, don't be puzzled lol, j-1 initially gives -1 which is consistent
    #     # draw on a piece of paper how this algorithm works
    #     if j < num_citations:
    #         if i > supertoken_range[j-1] and i < supertoken_range[j]:
    #             feature_mask[0][i] = j
    #         elif i == supertoken_range[j] and j < num_citations - 1:
    #             j += 1
    #             feature_mask[0][i] = j
    #         else:
    #             feature_mask[0][i] = 0
    #     else:
    #         feature_mask[0][i] = 0
    # feature_mask = feature_mask.to(inputs.device)
    print("feature_mask shape is", feature_mask.shape)
    print("feature_mask is", feature_mask)
    feature_mask_idxs = [0, 27, 203, 424, 572, 743, 919]
    for feature_mask_idx in feature_mask_idxs:
        print("feature_mask[0][{}] is".format(feature_mask_idx), feature_mask[0][feature_mask_idx])
    
    attributions = ig.attribute(
        inputs, # add batch dimension for Captum
        target=prediction_id,
        feature_mask=feature_mask,
        n_samples=300,
        show_progress=True
    ) # for [limebase]

    if decoder_ is not None:
        # Does it make sense to concatenate encoder and decoder attributions before normalization?
        # We assume that the encoder/decoder embeddings are the same
        normalized_attributes = normalize_attributes(torch.cat(attributions, dim=1))
    else:
        normalized_attributes = normalize_attributes(attributions)
        
    # print("normalized_attributes is", normalized_attributes)
    
    return normalized_attributes
