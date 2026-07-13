# Copyright (c) Alibaba, Inc. and its affiliates.

"""Runtime MCore patches for the DeltaNet RNN-SP1 naive baseline."""

from contextlib import nullcontext
from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor
from megatron.training import get_args

from .context import get_seq_split_context, set_seq_split_context

try:
    from swift.megatron.model.gpt_model import GPTModel as SwiftGPTModel
    from swift.megatron.model.rope import dynamic_rope_update
except Exception:  # pragma: no cover - optional outside Swift runtime.
    SwiftGPTModel = None
    dynamic_rope_update = None


_PATCHED = False


def _enabled() -> bool:
    try:
        args = get_args()
    except Exception:
        return False
    return bool(getattr(args, 'deltanet_rnn_sp1_baseline', False))


def _is_group(value) -> bool:
    return isinstance(value, list) and len(value) > 1


def _grouped_rotary_seq_len(decoder_inputs, input_ids):
    if _is_group(decoder_inputs):
        return sum(chunk.shape[0] for chunk in decoder_inputs)
    if _is_group(input_ids):
        return sum(chunk.shape[1] for chunk in input_ids)
    return None


def _decoder_input_tensor(model):
    decoder = getattr(model, 'decoder', None)
    return getattr(decoder, 'input_tensor', None)


def _split_first(output_with_bias, sizes):
    if isinstance(output_with_bias, tuple):
        output, *rest = output_with_bias
        return [(part, *rest) for part in torch.split(output, sizes, dim=0)]
    if isinstance(output_with_bias, list):
        output, *rest = output_with_bias
        return [[part, *rest] for part in torch.split(output, sizes, dim=0)]
    return list(torch.split(output_with_bias, sizes, dim=0))


def _call_attention_full_sequence(self, full_hidden_states, **kwargs):
    previous = get_seq_split_context()
    set_seq_split_context(0, 1, None, None)
    try:
        return self.self_attention(full_hidden_states, **kwargs)
    finally:
        set_seq_split_context(
            previous.micro_sp_idx,
            previous.pipe_sp_splits,
            previous.start,
            previous.end,
            microbatch_key=previous.microbatch_key,
        )


def _layer_forward_group(
    self,
    hidden_states,
    attention_mask: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    rotary_pos_cos: Optional[torch.Tensor] = None,
    rotary_pos_sin: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    inference_context=None,
    packed_seq_params=None,
    sequence_len_offset: Optional[torch.Tensor] = None,
    *,
    inference_params=None,
):
    inference_context = deprecate_inference_params(inference_context, inference_params)
    if inference_context is not None:
        raise RuntimeError('--deltanet-rnn-sp1-baseline does not support inference context')
    if packed_seq_params is not None:
        raise RuntimeError('--deltanet-rnn-sp1-baseline does not support packed_seq_params')
    if self.recompute_input_layernorm or self.recompute_pre_mlp_layernorm or self.recompute_mlp:
        raise RuntimeError('--deltanet-rnn-sp1-baseline does not support selective recompute')

    residuals = hidden_states
    input_layernorm_outputs = [self.input_layernorm(chunk) for chunk in hidden_states]
    sizes = [chunk.size(0) for chunk in input_layernorm_outputs]
    full_input = torch.cat(input_layernorm_outputs, dim=0)

    attention_output_with_bias = _call_attention_full_sequence(
        self,
        full_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        attention_bias=attention_bias,
        packed_seq_params=None,
        sequence_len_offset=sequence_len_offset,
    )
    attention_chunks = _split_first(attention_output_with_bias, sizes)

    hidden_chunks = []
    for attention_chunk, residual in zip(attention_chunks, residuals):
        with self.bias_dropout_add_exec_handler():
            hidden_chunks.append(
                self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                    attention_chunk, residual, self.hidden_dropout
                )
            )

    pre_mlp_chunks = []
    residual_chunks = []
    for chunk in hidden_chunks:
        residual = chunk
        pre_cross = self.pre_cross_attn_layernorm(chunk)
        cross_output_with_bias = self.cross_attention(
            pre_cross,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )
        if isinstance(cross_output_with_bias, dict) and 'context' in cross_output_with_bias:
            context = cross_output_with_bias['context']
        with self.bias_dropout_add_exec_handler():
            chunk = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                cross_output_with_bias, residual, self.hidden_dropout
            )
        residual_chunks.append(chunk)
        pre_mlp_chunks.append(self.pre_mlp_layernorm(chunk))

    outputs = [
        self._forward_mlp(pre_mlp, residual)
        for pre_mlp, residual in zip(pre_mlp_chunks, residual_chunks)
    ]
    return outputs, context


def apply_rnn_sp1_model_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    GPTModel._rnn_sp1_original_set_input_tensor = GPTModel.set_input_tensor
    GPTModel._rnn_sp1_original_forward = GPTModel.forward
    if SwiftGPTModel is not None:
        SwiftGPTModel._rnn_sp1_original_forward = SwiftGPTModel.forward
    TransformerBlock._rnn_sp1_original_forward = TransformerBlock.forward
    TransformerLayer._rnn_sp1_original_forward = TransformerLayer.forward

    def set_input_tensor(self, input_tensor):
        if _enabled() and _is_group(input_tensor):
            self.decoder.set_input_tensor(input_tensor)
            return
        return self._rnn_sp1_original_set_input_tensor(input_tensor)

    def gpt_forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output=None,
        *,
        inference_params=None,
        loss_mask=None,
    ):
        decoder_input_tensor = _decoder_input_tensor(self)
        grouped = _is_group(input_ids) or _is_group(decoder_input) or _is_group(decoder_input_tensor)
        if not (_enabled() and grouped):
            return self._rnn_sp1_original_forward(
                input_ids,
                position_ids,
                attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
                inference_params=inference_params,
                loss_mask=loss_mask,
            )
        if packed_seq_params is not None:
            raise RuntimeError('--deltanet-rnn-sp1-baseline does not support packed_seq_params')

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if decoder_input is not None:
            decoder_inputs = decoder_input
        elif self.pre_process:
            decoder_inputs = [
                self.embedding(input_ids=ids, position_ids=pos)
                for ids, pos in zip(input_ids, position_ids)
            ]
        else:
            decoder_inputs = None

        hidden_states = self.decoder(
            hidden_states=decoder_inputs,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        output_weight = self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else None
        logits = [
            self.output_layer(chunk, weight=output_weight, runtime_gather_output=runtime_gather_output)[0]
            for chunk in hidden_states
        ]

        if labels is None:
            return [chunk.transpose(0, 1).contiguous() for chunk in logits]

        return [
            self.compute_language_model_loss(label, logit)
            for label, logit in zip(labels, logits)
        ]

    def swift_gpt_forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_params=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output=None,
    ):
        decoder_input_tensor = _decoder_input_tensor(self)
        grouped = _is_group(input_ids) or _is_group(decoder_input) or _is_group(decoder_input_tensor)
        if not (_enabled() and grouped):
            return self._rnn_sp1_original_forward(
                input_ids,
                position_ids,
                attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                runtime_gather_output=runtime_gather_output,
            )
        if packed_seq_params is not None:
            raise RuntimeError('--deltanet-rnn-sp1-baseline does not support packed_seq_params')

        if decoder_input is not None:
            decoder_inputs = decoder_input
        elif self.pre_process:
            decoder_inputs = [
                self.embedding(input_ids=ids, position_ids=pos)
                for ids, pos in zip(input_ids, position_ids)
            ]
        else:
            decoder_inputs = None

        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        sequence_len_offset = None
        if getattr(self, 'position_embedding_type', None) == 'rope':
            rotary_seq_len = _grouped_rotary_seq_len(decoder_inputs, input_ids)
            if rotary_seq_len is None and _is_group(decoder_input_tensor):
                rotary_seq_len = sum(chunk.shape[0] for chunk in decoder_input_tensor)
            if rotary_seq_len is None:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_params,
                    self.decoder,
                    decoder_input,
                    self.config,
                    packed_seq_params,
                )
            if getattr(self, 'hf_rope_scaling', None) is not None and dynamic_rope_update is not None:
                attention_scaling = dynamic_rope_update(self, self.rotary_pos_emb.inv_freq, rotary_seq_len)
                if attention_scaling is not None:
                    self.attention_scaling = attention_scaling
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        with self._patch_apply_rotary_pos_emb():
            hidden_states = self.decoder(
                hidden_states=decoder_inputs,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=None,
                sequence_len_offset=sequence_len_offset,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        output_weight = self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else None
        logits = [
            self.output_layer(chunk, weight=output_weight, runtime_gather_output=runtime_gather_output)[0]
            for chunk in hidden_states
        ]

        if labels is None:
            return [chunk.transpose(0, 1).contiguous() for chunk in logits]

        return [
            self.compute_language_model_loss(label, logit)
            for label, logit in zip(labels, logits)
        ]

    def block_forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        grouped = _is_group(hidden_states) or (not self.pre_process and _is_group(self.input_tensor))
        if not (_enabled() and grouped):
            return self._rnn_sp1_original_forward(
                hidden_states,
                attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        if not self.pre_process:
            hidden_states = self.input_tensor
        if self.config.recompute_granularity == 'full':
            if (
                getattr(self.config, 'recompute_method', None) != 'uniform'
                or getattr(self.config, 'recompute_num_layers', None) != 1
            ):
                raise RuntimeError(
                    '--deltanet-rnn-sp1-baseline full recompute currently supports '
                    'only recompute_method=uniform and recompute_num_layers=1')
            if getattr(self.config, 'distribute_saved_activations', False):
                raise RuntimeError(
                    '--deltanet-rnn-sp1-baseline full recompute does not support '
                    'distribute_saved_activations')
        if inference_context is not None:
            raise RuntimeError('--deltanet-rnn-sp1-baseline does not support inference context')

        hidden_states = [
            make_viewless_tensor(inp=chunk, requires_grad=True, keep_graph=True)
            for chunk in hidden_states
        ]

        rng_context = tensor_parallel.get_cuda_rng_tracker().fork() if self.config.sequence_parallel else nullcontext()
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        use_full_recompute = (
            self.config.recompute_granularity == 'full'
            and self.training
            and torch.is_grad_enabled()
        )

        def run_grouped_layer(layer, chunks):
            return layer(
                hidden_states=list(chunks),
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=None,
                sequence_len_offset=sequence_len_offset,
            )

        def checkpoint_grouped_layer(layer, chunks):
            def custom_forward(*chunk_tensors):
                outputs, _ = run_grouped_layer(layer, chunk_tensors)
                return tuple(outputs)

            outputs = tensor_parallel.checkpoint(
                custom_forward,
                getattr(self.config, 'distribute_saved_activations', False),
                *chunks)
            return list(outputs), context

        with rng_context, outer_fp8_context:
            for layer in self.layers:
                inner_fp8_context = (
                    get_fp8_context(self.config, layer.layer_number - 1)
                    if use_inner_fp8_context
                    else nullcontext()
                )
                with self.offload_context, inner_fp8_context:
                    if use_full_recompute:
                        hidden_states, context = checkpoint_grouped_layer(layer, hidden_states)
                    else:
                        hidden_states, context = run_grouped_layer(layer, hidden_states)

        if self.final_layernorm is not None:
            hidden_states = [
                make_viewless_tensor(
                    inp=self.final_layernorm(chunk),
                    requires_grad=True,
                    keep_graph=True,
                )
                for chunk in hidden_states
            ]

        return hidden_states

    def layer_forward(self, *args, **kwargs):
        hidden_states = kwargs.get('hidden_states', args[0] if args else None)
        if _enabled() and _is_group(hidden_states):
            return _layer_forward_group(self, *args, **kwargs)
        return self._rnn_sp1_original_forward(*args, **kwargs)

    GPTModel.set_input_tensor = set_input_tensor
    GPTModel.forward = gpt_forward
    if SwiftGPTModel is not None:
        SwiftGPTModel.forward = swift_gpt_forward
    TransformerBlock.forward = block_forward
    TransformerLayer.forward = layer_forward
