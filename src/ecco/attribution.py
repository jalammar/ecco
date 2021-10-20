import torch
from typing import Optional


def saliency(prediction_logit, encoder_token_ids_tensor_one_hot, decoder_token_ids_tensor_one_hot: Optional = None,
             norm=True, retain_graph=False):

    if decoder_token_ids_tensor_one_hot is not None:
        # only works in batches of 1
        assert len(decoder_token_ids_tensor_one_hot.shape) == 3 and decoder_token_ids_tensor_one_hot.shape[0] == 1
        assert len(encoder_token_ids_tensor_one_hot.shape) == 3 and encoder_token_ids_tensor_one_hot.shape[0] == 1
    else:
        # only works for vector without batch dimension
        assert len(encoder_token_ids_tensor_one_hot.shape) == 2

    # Back-propegate the gradient from the selected output-logit
    prediction_logit.backward(retain_graph=retain_graph)

    token_ids_tensor_one_hot_grad = torch.cat(
        [encoder_token_ids_tensor_one_hot.grad, decoder_token_ids_tensor_one_hot.grad], dim=1
    )[0] if decoder_token_ids_tensor_one_hot is not None else encoder_token_ids_tensor_one_hot.grad

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


def saliency_on_d_embeddings(prediction_logit, inputs_embeds, aggregation="L2", retain_graph=True):
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
                                  retain_graph=True):

    if decoder_inputs_embeds is not None:
        # only works in batches of 1
        assert len(decoder_inputs_embeds.shape) == 3 and decoder_inputs_embeds.shape[0] == 1
        assert len(encoder_inputs_embeds.shape) == 3 and encoder_inputs_embeds.shape[0] == 1

        decoder_inputs_embeds.retain_grad()

    encoder_inputs_embeds.retain_grad()

    # back-prop gradient
    prediction_logit.backward(retain_graph=retain_graph)
    decoder_grad = decoder_inputs_embeds.grad if decoder_inputs_embeds is not None else None
    encoder_grad = encoder_inputs_embeds.grad
    # This should be equivalent to
    # grad = torch.autograd.grad(prediction_logit, inputs_embeds)[0]

    # Grad X Input
    grad_enc_x_input = encoder_grad * encoder_inputs_embeds

    if decoder_grad is not None:
        grad_dec_x_input = decoder_grad * decoder_inputs_embeds
        grad_enc_x_input = encoder_grad * encoder_inputs_embeds
        grad_x_input = torch.cat([grad_enc_x_input, grad_dec_x_input], dim=1)[0]
    else:
        grad_x_input = grad_enc_x_input

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


def compute_saliency_scores(prediction_logit,
                            encoder_token_ids_tensor_one_hot,
                            encoder_inputs_embeds,
                            decoder_token_ids_tensor_one_hot: Optional = None,
                            decoder_inputs_embeds: Optional = None,
                            gradient_kwargs={},
                            gradient_x_input_kwargs={}):

    results = {}

    results['grad_x_input'] = gradient_x_inputs_attribution(
        prediction_logit=prediction_logit,
        encoder_inputs_embeds=encoder_inputs_embeds,
        decoder_inputs_embeds=decoder_inputs_embeds,
        **gradient_x_input_kwargs
    )

    results['gradient'] = saliency(
        prediction_logit=prediction_logit,
        encoder_token_ids_tensor_one_hot=encoder_token_ids_tensor_one_hot,
        decoder_token_ids_tensor_one_hot=decoder_token_ids_tensor_one_hot,
        **gradient_kwargs
    )

    return results
