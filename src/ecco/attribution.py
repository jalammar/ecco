from functools import partial
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
    LRP
)
from torch.nn import functional as F
import transformers


ATTR_NAME_ALIASES = {
    'ig': 'integrated_gradients',
    'saliency': 'gradient',
    'dl': 'deep_lift',
    'dls': 'deep_lift_shap',
    'gb': 'guided_backprop',
    'gg': 'guided_gradcam',
    'deconv': 'deconvolution',
    'lrp': 'layer_relevance_propagation'
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
    'layer_relevance_propagation': LRP
}


def compute_primary_attributions_scores(attr_method : str, model: transformers.PreTrainedModel,
                                        forward_kwargs: Dict[str, Any], prediction_id: torch.Tensor,
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

    if decoder_ is None:
        forward_func = partial(model_forward, decoder_=decoder_, model=model, extra_forward_args=extra_forward_args)
        inputs = input_
    else:
        forward_func = partial(model_forward, model=model, extra_forward_args=extra_forward_args)
        inputs = tuple([input_, decoder_])

    attr_method_class = ATTR_NAME_TO_CLASS.get(ATTR_NAME_ALIASES.get(attr_method, attr_method), None)
    if attr_method_class is None:
        raise NotImplementedError(
            f"No implementation found for primary attribution method '{attr_method}'. "
            f"Please choose one of the methods: {list(ATTR_NAME_TO_CLASS.keys())}"
        )

    ig = attr_method_class(forward_func=forward_func)
    attributions = ig.attribute(inputs, target=prediction_id)

    if decoder_ is not None:
        # Does it make sense to concatenate encoder and decoder attributions before normalization?
        # We assume that the encoder/decoder embeddings are the same
        return normalize_attributes(torch.cat(attributions, dim=1))
    else:
        return normalize_attributes(attributions)
