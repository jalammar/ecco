from functools import partial
import torch
from typing import Any, Dict, Optional, Tuple, List
from captum.attr import IntegratedGradients
from torch.nn import functional as F
import transformers


def saliency(prediction_logit, encoder_token_ids_tensor_one_hot, decoder_token_ids_tensor_one_hot: Optional = None,
             norm=True, retain_graph=False) -> torch.Tensor:

    # only works in batches of 1
    assert len(encoder_token_ids_tensor_one_hot.shape) == 3 and encoder_token_ids_tensor_one_hot.shape[0] == 1
    if decoder_token_ids_tensor_one_hot is not None:
        assert len(decoder_token_ids_tensor_one_hot.shape) == 3 and decoder_token_ids_tensor_one_hot.shape[0] == 1

    # Back-propegate the gradient from the selected output-logit
    prediction_logit.backward(retain_graph=retain_graph)

    token_ids_tensor_one_hot_grad = torch.cat(
        [encoder_token_ids_tensor_one_hot.grad, decoder_token_ids_tensor_one_hot.grad], dim=1
    )[0] if decoder_token_ids_tensor_one_hot is not None else encoder_token_ids_tensor_one_hot.grad[0]

    # token_ids_tensor_one_hot.grad is the gradient propegated to ever embedding dimension of
    # the input tokens.
    if norm:  # norm calculates a scalar value (L2 Norm)
        token_importance_raw = torch.norm(token_ids_tensor_one_hot_grad, dim=1)
        # print('token_importance_raw', token_ids_tensor_one_hot.grad.shape,
        # np.count_nonzero(token_ids_tensor_one_hot.detach().numpy(), axis=1))

        # Normalize the values so they add up to 1
        token_importance = token_importance_raw / torch.sum(token_importance_raw)
    else:
        token_importance = torch.sum(token_ids_tensor_one_hot_grad, dim=1)  # Only one value, all others are zero

    encoder_token_ids_tensor_one_hot.grad.data.zero_()
    if decoder_token_ids_tensor_one_hot is not None:
        decoder_token_ids_tensor_one_hot.grad.data.zero_()

    return token_importance


def saliency_on_d_embeddings(prediction_logit, inputs_embeds, aggregation="L2", retain_graph=True) -> torch.Tensor:
    inputs_embeds.retain_grad()

    # Back-propegate the gradient from the selected output-logit
    prediction_logit.backward(retain_graph=retain_graph)

    # inputs_embeds.grad
    # token_ids_tensor_one_hot.grad is the gradient propegated to ever embedding dimension of
    # the input tokens.
    if aggregation == "L2":  # norm calculates a scalar value (L2 Norm)
        token_importance_raw = torch.norm(inputs_embeds.grad, dim=1)
        # print('token_importance_raw', token_ids_tensor_one_hot.grad.shape,
        # np.count_nonzero(token_ids_tensor_one_hot.detach().numpy(), axis=1))

        # Normalize the values so they add up to 1
        token_importance = token_importance_raw / torch.sum(token_importance_raw)
    elif aggregation == "sum":
        token_importance_raw = torch.sum(inputs_embeds.grad, dim=1)
        token_importance = token_importance_raw  # Hmmm, how to normalize if it includes negative values
    elif aggregation == "mean":
        token_importance_raw = torch.mean(inputs_embeds.grad, dim=1)
        token_importance = token_importance_raw  # Hmmm, how to normalize if it includes negative values

    inputs_embeds.grad.data.zero_()
    return token_importance


def gradient_x_inputs_attribution(prediction_logit, encoder_inputs_embeds, decoder_inputs_embeds: Optional = None,
                                  retain_graph=True) -> torch.Tensor:

    # only works in batches of 1
    assert len(encoder_inputs_embeds.shape) == 3 and encoder_inputs_embeds.shape[0] == 1
    if decoder_inputs_embeds is not None:
        assert len(decoder_inputs_embeds.shape) == 3 and decoder_inputs_embeds.shape[0] == 1
        decoder_inputs_embeds.retain_grad()
    encoder_inputs_embeds.retain_grad()

    # back-prop gradient
    prediction_logit.backward(retain_graph=retain_graph)
    decoder_grad = decoder_inputs_embeds.grad if decoder_inputs_embeds is not None else None
    encoder_grad = encoder_inputs_embeds.grad

    # Grad X Input
    grad_enc_x_input = encoder_grad * encoder_inputs_embeds

    if decoder_grad is not None:
        grad_dec_x_input = decoder_grad * decoder_inputs_embeds
        grad_enc_x_input = encoder_grad * encoder_inputs_embeds
        grad_x_input = torch.cat([grad_enc_x_input, grad_dec_x_input], dim=1)[0]
    else:
        grad_x_input = grad_enc_x_input[0]

    # Turn into a scalar value for each input token by taking L2 norm
    feature_importance = torch.norm(grad_x_input, dim=1)

    # Normalize so we can show scores as percentages
    token_importance_normalized = feature_importance / torch.sum(feature_importance)

    # Zero the gradient for the tensor so next backward() calls don't have
    # gradients accumulating
    if decoder_inputs_embeds is not None:
        decoder_inputs_embeds.grad.data.zero_()
    encoder_inputs_embeds.grad.data.zero_()

    return token_importance_normalized


def compute_integrated_gradients_scores(model: transformers.PreTrainedModel, forward_kwargs: Dict[str, Any],
                                        prediction_id: torch.Tensor, aggregation: str = "L2") -> torch.Tensor:

    """
    Computes the Integrated Gradients primary attributions with respect to the specified `prediction_id`.

    Args:
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
            attributes = norm / torch.sum(norm) # Normalize the values so they add up to 1
        else:
            raise NotImplemented

        return attributes

    extra_forward_args = {k: v for k, v in forward_kwargs.items() if k not in ['inputs_embeds', 'decoder_inputs_embeds']}
    input_ = forward_kwargs.get('inputs_embeds')
    decoder_ = forward_kwargs.get('decoder_inputs_embeds')

    if decoder_ is None:
        forward_func = partial(model_forward, decoder_=decoder_, model=model, extra_forward_args=extra_forward_args)
        inputs = input_
    else:
        forward_func = partial(model_forward, model=model, extra_forward_args=extra_forward_args)
        inputs = tuple([input_, decoder_])

    ig = IntegratedGradients(forward_func=forward_func)
    attributions = ig.attribute(inputs, target=prediction_id)
     
    if decoder_ is not None:
        # Does it make sense to concatenate encoder and decoder attributions before normalization?
        # We assume that the encoder/decoder embeddings are the same
        return normalize_attributes(torch.cat(attributions, dim=1))
    else:
        return normalize_attributes(attributions)


def compute_saliency_scores(prediction_logit: torch.Tensor,
                            encoder_token_ids_tensor_one_hot: torch.Tensor,
                            encoder_inputs_embeds: torch.Tensor,
                            decoder_token_ids_tensor_one_hot: Optional[torch.Tensor] = None,
                            decoder_inputs_embeds: Optional[torch.Tensor] = None,
                            gradient_kwargs: Dict[str, Any] = {},
                            gradient_x_input_kwargs: Dict[str, Any] = {},
                            saliency_methods: Optional[List[str]] = ['grad_x_input', 'gradient']) \
        -> Dict[str, torch.Tensor]:

    results = {}

    if 'grad_x_input' in saliency_methods:
        results['grad_x_input'] = gradient_x_inputs_attribution(
            prediction_logit=prediction_logit,
            encoder_inputs_embeds=encoder_inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **gradient_x_input_kwargs
        )


    if 'gradient' in saliency_methods:
        results['gradient'] = saliency(
            prediction_logit=prediction_logit,
            encoder_token_ids_tensor_one_hot=encoder_token_ids_tensor_one_hot,
            decoder_token_ids_tensor_one_hot=decoder_token_ids_tensor_one_hot,
            **gradient_kwargs
        )

    return results