# Copyright (c) Alibaba, Inc. and its affiliates.

"""Seq1F1B schedule patch for DeltaNet on Megatron-Core r0.12 non-interleaved PP."""

import contextlib
from typing import Iterator, List, Union

import torch
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import schedules as mcore_schedules
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.utils import get_model_config, get_model_type, get_model_xattn
from megatron.training import get_args

from .context import reset_split_progress


class SeqSplitQueue:
    """FIFO across microbatches, LIFO across sequence splits of one microbatch."""

    def __init__(self, pipe_sp_splits: int):
        self.queues = [[]]
        self.pipe_sp_splits = pipe_sp_splits
        self.offset = 0
        self.idx = 0
        self.count = 0

    def append(self, obj):
        self.queues[self.offset].append(obj)
        self.idx += 1
        if self.idx == self.pipe_sp_splits:
            self.queues.append([])
            self.idx = 0
            self.offset += 1
        self.count += 1

    def pop(self, idx=0):
        assert idx == 0
        self.count -= 1
        if len(self.queues[0]) == 1:
            if self.offset > 0:
                self.offset -= 1
                return self.queues.pop(0)[0]
            return self.queues[0].pop(-1)
        return self.queues[0].pop(-1)


def forward_backward_pipelining_without_interleaving_seq1f1b(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B with Seq1F1B sequence splits.

    This mirrors Megatron-Core r0.12's non-interleaved schedule but treats each
    original microbatch as ``pipe_sp_splits`` smaller sequence chunks. Backward
    uses ``SeqSplitQueue`` so chunks inside one original sequence are popped in
    reverse order, which is required by DeltaNet's manual state-gradient relay.
    """

    args = get_args()
    pipe_sp_splits = getattr(args, 'pipe_sp_splits', 1)
    assert pipe_sp_splits > 1
    assert seq_length % pipe_sp_splits == 0
    # Every forward-backward run must start from chunk 0 of a fresh batch,
    # otherwise the data slicing in the trainer would desync from the schedule.
    reset_split_progress()
    local_seq_length = seq_length // pipe_sp_splits
    total_num_microbatches = num_microbatches * pipe_sp_splits

    if isinstance(model, list):
        assert len(model) == 1, 'Seq1F1B patch currently supports non-interleaved PP only.'
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, 'Seq1F1B patch currently supports one data iterator.'
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError('Seq1F1B non-interleaved patch does not support overlapping p2p communication.')

    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = mcore_schedules.clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func or contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    num_warmup_microbatches = pp_size - rank + pipe_sp_splits - 2
    num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)
    recv_tensor_shapes = mcore_schedules.get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=local_seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = mcore_schedules.get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=local_seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()
    if not forward_only:
        input_tensors = SeqSplitQueue(pipe_sp_splits)
        output_tensors = SeqSplitQueue(pipe_sp_splits)
    forward_data_store = []

    for i in range(num_warmup_microbatches):
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = mcore_schedules.recv_forward(recv_tensor_shapes, config)
        output_tensor, num_tokens = mcore_schedules.forward_step(
            forward_step_func,
            data_iterator,
            model,
            total_num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            mcore_schedules.check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        mcore_schedules.send_forward(output_tensor, send_tensor_shapes, config)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            mcore_schedules.deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    if num_microbatches_remaining > 0:
        input_tensor = mcore_schedules.recv_forward(recv_tensor_shapes, config)

    for i in range(num_microbatches_remaining):
        virtual_microbatch = i + num_warmup_microbatches
        last_iteration = i == (num_microbatches_remaining - 1)

        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                virtual_microbatch % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = mcore_schedules.forward_step(
            forward_step_func,
            data_iterator,
            model,
            total_num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            mcore_schedules.check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=virtual_microbatch,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        total_num_tokens += num_tokens

        if forward_only:
            mcore_schedules.send_forward(output_tensor, send_tensor_shapes, config)
            if not last_iteration:
                input_tensor = mcore_schedules.recv_forward(recv_tensor_shapes, config)
        else:
            output_tensor_grad = mcore_schedules.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            mcore_schedules.deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = mcore_schedules.backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                mcore_schedules.send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                input_tensor = mcore_schedules.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )

    if not forward_only:
        for i in range(num_warmup_microbatches):
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            output_tensor_grad = mcore_schedules.recv_backward(send_tensor_shapes, config)
            input_tensor_grad = mcore_schedules.backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            mcore_schedules.send_backward(input_tensor_grad, recv_tensor_shapes, config)

        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:
        mcore_schedules.finish_embedding_wgrad_compute(config, embedding_module)
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store

