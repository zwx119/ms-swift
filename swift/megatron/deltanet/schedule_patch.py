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


def _disable_te_first_microbatch_optimization(model):
    """Disable TE first-microbatch grad shortcut for LIFO chunk backward.

    TransformerEngine uses ``is_first_microbatch`` to skip gradient accumulation
    for the first microbatch. Seq1F1B backpropagates chunks of one sequence in
    reverse order, so the first forward chunk is not the first chunk to reach
    backward. Leaving the shortcut enabled lets chunk0 overwrite gradients that
    chunks 1..N already accumulated.
    """

    modules = model if isinstance(model, list) else [model]
    for module in modules:
        if not hasattr(module, "modules"):
            continue
        for submodule in module.modules():
            if hasattr(submodule, "is_first_microbatch"):
                submodule.is_first_microbatch = False
            if hasattr(submodule, "disable_parameter_transpose_cache"):
                submodule.disable_parameter_transpose_cache = True


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


def get_seq1f1b_splits(args=None, seq_length=None) -> List[int]:
    """Return the active Seq1F1B split lengths.

    Swift's alpha sweep drives non-uniform candidates through
    ``pipe_sp_strategy=manual``. Keeping the parser here lets the trainer and
    pipeline schedule share exactly the same split list.
    """

    args = get_args() if args is None else args
    pipe_sp_splits = int(getattr(args, 'pipe_sp_splits', 1))
    if seq_length is None:
        seq_length = int(getattr(args, 'seq_length'))
    else:
        seq_length = int(seq_length)
    if pipe_sp_splits <= 1:
        return [seq_length]

    strategy = getattr(args, 'pipe_sp_strategy', 'average')
    if strategy == 'average':
        if seq_length % pipe_sp_splits != 0:
            raise ValueError(
                f'Average Seq1F1B requires seq_length={seq_length} to be divisible '
                f'by pipe_sp_splits={pipe_sp_splits}.'
            )
        return [seq_length // pipe_sp_splits] * pipe_sp_splits

    if strategy != 'manual':
        raise ValueError(f'Unsupported DeltaNet pipe_sp_strategy for Swift: {strategy}')

    spec = getattr(args, 'pipe_sp_manual_splits', '')
    splits = [int(item.strip()) for item in spec.split(',') if item.strip()]
    if len(splits) != pipe_sp_splits:
        raise ValueError(
            f'Expected {pipe_sp_splits} manual splits, got {len(splits)}: {splits}'
        )
    if any(split <= 0 for split in splits):
        raise ValueError(f'Manual splits must be positive: {splits}')
    if sum(splits) != seq_length:
        raise ValueError(
            f'Manual splits must sum to seq_length={seq_length}, got '
            f'{sum(splits)}: {splits}'
        )
    if seq_length % 128 == 0 and any(split % 128 != 0 for split in splits):
        raise ValueError(f'Manual splits must be multiples of 128: {splits}')
    return splits


def _tensor_shapes_for_length(rank, model_type, seq_length, micro_batch_size,
                              decoder_seq_length, config, encoder_decoder_xattn):
    return mcore_schedules.get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )


def _recv_forward_group(tensor_shapes, config, group_size):
    if parallel_state.is_pipeline_first_stage():
        return None
    return [mcore_schedules.recv_forward(tensor_shapes, config)[0] for _ in range(group_size)]


def _send_forward_group(output_tensors, tensor_shapes, config):
    if parallel_state.is_pipeline_last_stage():
        return
    for output_tensor in output_tensors:
        mcore_schedules.send_forward([output_tensor], tensor_shapes, config)


def _recv_backward_group(tensor_shapes, config, group_size):
    if parallel_state.is_pipeline_last_stage():
        return None
    return [mcore_schedules.recv_backward(tensor_shapes, config)[0] for _ in range(group_size)]


def _send_backward_group(input_tensor_grads, tensor_shapes, config):
    if parallel_state.is_pipeline_first_stage():
        return
    for input_tensor_grad in input_tensor_grads:
        mcore_schedules.send_backward([input_tensor_grad], tensor_shapes, config)


def _send_forward_recv_backward_group(output_tensors, tensor_shapes, config):
    if parallel_state.is_pipeline_last_stage():
        return None
    grads = []
    for output_tensor in output_tensors:
        grad = mcore_schedules.send_forward_recv_backward([output_tensor], tensor_shapes, config)
        grads.append(grad[0])
    return grads


def _send_backward_recv_forward_group(input_tensor_grads, tensor_shapes, config, group_size):
    if parallel_state.is_pipeline_first_stage():
        return _recv_forward_group(tensor_shapes, config, group_size)
    inputs = []
    for input_tensor_grad in input_tensor_grads:
        input_tensor = mcore_schedules.send_backward_recv_forward([input_tensor_grad], tensor_shapes, config)
        inputs.append(input_tensor[0])
    return inputs


def _backward_step_group(input_tensors, output_tensors, output_tensor_grads, model_type, config):
    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    input_was_none = input_tensors is None
    input_tensors = [] if input_tensors is None else input_tensors
    for tensor in input_tensors:
        if tensor is not None:
            tensor.retain_grad()

    output_is_group = isinstance(output_tensors, list)
    if not output_is_group:
        output_tensors = [output_tensors]
    if output_tensor_grads is None:
        output_tensor_grads = [None] * len(output_tensors)
    elif not isinstance(output_tensor_grads, list):
        output_tensor_grads = [output_tensor_grads]

    if output_tensor_grads[0] is None and config.grad_scale_func is not None:
        output_tensors[0] = config.grad_scale_func(output_tensors[0])

    if len(output_tensors) == 1 and output_tensor_grads[0] is None:
        torch.autograd.backward(output_tensors[0])
    else:
        torch.autograd.backward(output_tensors, grad_tensors=output_tensor_grads)

    input_tensor_grads = None if input_was_none else [
        None if tensor is None else tensor.grad
        for tensor in input_tensors
    ]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grads


def _forward_step_group(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensors,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
):
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        mcore_schedules.set_current_microbatch(model, current_microbatch)

    set_input_tensor = mcore_schedules.get_attr_wrapped_model(model, 'set_input_tensor')
    set_input_tensor([None] if input_tensors is None else input_tensors)

    context_manager = (
        torch.autocast('cuda', dtype=config.autocast_dtype)
        if config.enable_autocast
        else contextlib.nullcontext()
    )
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensors, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensors, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int, device=torch.cuda.current_device())
    if parallel_state.is_pipeline_last_stage():
        if collect_non_loss_data:
            data = loss_func(output_tensors, non_loss_data=True)
            forward_data_store.append(data)
        else:
            outputs = loss_func(output_tensors)
            if len(outputs) == 3:
                output_tensors, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensors /= num_tokens
                    output_tensors *= parallel_state.get_context_parallel_world_size()
                    output_tensors /= num_microbatches
            else:
                output_tensors, loss_reduced = outputs
                output_tensors *= parallel_state.get_context_parallel_world_size()
                output_tensors /= num_microbatches
            forward_data_store.append(loss_reduced)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    return output_tensors, num_tokens


def forward_backward_pipelining_without_interleaving_rnn_sp1(
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
    """Grouped non-interleaved PP for the DeltaNet RNN-SP1 naive baseline."""

    args = get_args()
    pipe_sp_splits = getattr(args, 'pipe_sp_splits', 1)
    assert pipe_sp_splits > 1
    if getattr(args, 'pipe_sp_strategy', 'average') != 'average':
        raise ValueError('RNN-SP1 non-interleaved patch currently supports only average Seq1F1B splits.')
    assert seq_length % pipe_sp_splits == 0
    reset_split_progress()
    local_seq_length = seq_length // pipe_sp_splits

    if isinstance(model, list):
        assert len(model) == 1, 'RNN-SP1 patch currently supports non-interleaved PP only.'
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, 'RNN-SP1 patch currently supports one data iterator.'
        data_iterator = data_iterator[0]

    _disable_te_first_microbatch_optimization(model)

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError('RNN-SP1 non-interleaved patch does not support overlapping p2p communication.')
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
    num_warmup_microbatches = pp_size - rank - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

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

    input_tensors = []
    output_tensors = []
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()
    forward_data_store = []

    for i in range(num_warmup_microbatches):
        checkpoint_activations_microbatch = None
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )

        input_group = _recv_forward_group(recv_tensor_shapes, config, pipe_sp_splits)
        output_group, num_tokens = _forward_step_group(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_group,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            mcore_schedules.check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
        )
        _send_forward_group(output_group, send_tensor_shapes, config)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_group)
            output_tensors.append(output_group)

    input_group = None
    if num_microbatches_remaining > 0:
        input_group = _recv_forward_group(recv_tensor_shapes, config, pipe_sp_splits)

    num_backwards_done = 0
    for i in range(num_microbatches_remaining):
        microbatch_id = i + num_warmup_microbatches
        last_iteration = i == (num_microbatches_remaining - 1)
        checkpoint_activations_microbatch = None
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                microbatch_id % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints

        output_group, num_tokens = _forward_step_group(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_group,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            mcore_schedules.check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=microbatch_id,
        )
        total_num_tokens += num_tokens

        if forward_only:
            _send_forward_group(output_group, send_tensor_shapes, config)
            if not last_iteration:
                input_group = _recv_forward_group(recv_tensor_shapes, config, pipe_sp_splits)
            continue

        output_group_grad = _send_forward_recv_backward_group(output_group, send_tensor_shapes, config)
        input_tensors.append(input_group)
        output_tensors.append(output_group)

        input_group_to_backward = input_tensors.pop(0)
        output_group_to_backward = output_tensors.pop(0)

        if num_warmup_microbatches == 0 and last_iteration:
            if config.grad_sync_func is None or rank == 0:
                enable_grad_sync()

        input_group_grad = _backward_step_group(
            input_group_to_backward,
            output_group_to_backward,
            output_group_grad,
            model_type,
            config,
        )
        num_backwards_done += 1

        if last_iteration:
            input_group = None
            _send_backward_group(input_group_grad, recv_tensor_shapes, config)
        else:
            input_group = _send_backward_recv_forward_group(
                input_group_grad, recv_tensor_shapes, config, pipe_sp_splits
            )

    if not forward_only:
        for i in range(num_warmup_microbatches):
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            output_group_grad = _recv_backward_group(send_tensor_shapes, config, pipe_sp_splits)
            input_group_to_backward = input_tensors.pop(0)
            output_group_to_backward = output_tensors.pop(0)
            input_group_grad = _backward_step_group(
                input_group_to_backward,
                output_group_to_backward,
                output_group_grad,
                model_type,
                config,
            )
            num_backwards_done += 1
            _send_backward_group(input_group_grad, recv_tensor_shapes, config)

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
    splits = get_seq1f1b_splits(args, seq_length)
    assert len(splits) == pipe_sp_splits
    # Every forward-backward run must start from chunk 0 of a fresh batch,
    # otherwise the data slicing in the trainer would desync from the schedule.
    reset_split_progress()
    total_num_microbatches = num_microbatches * pipe_sp_splits

    if isinstance(model, list):
        assert len(model) == 1, 'Seq1F1B patch currently supports non-interleaved PP only.'
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, 'Seq1F1B patch currently supports one data iterator.'
        data_iterator = data_iterator[0]

    _disable_te_first_microbatch_optimization(model)

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

    def recv_shapes_for_virtual(virtual_microbatch: int):
        return _tensor_shapes_for_length(
            rank=rank - 1,
            model_type=model_type,
            seq_length=splits[virtual_microbatch % pipe_sp_splits],
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )

    def send_shapes_for_virtual(virtual_microbatch: int):
        return _tensor_shapes_for_length(
            rank=rank,
            model_type=model_type,
            seq_length=splits[virtual_microbatch % pipe_sp_splits],
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )

    input_tensors = None
    output_tensors = None
    output_tensor_shapes = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()
    if not forward_only:
        input_tensors = SeqSplitQueue(pipe_sp_splits)
        output_tensors = SeqSplitQueue(pipe_sp_splits)
        output_tensor_shapes = SeqSplitQueue(pipe_sp_splits)
    forward_data_store = []

    for i in range(num_warmup_microbatches):
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = mcore_schedules.recv_forward(recv_shapes_for_virtual(i), config)
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
        output_shape = send_shapes_for_virtual(i)
        mcore_schedules.send_forward(output_tensor, output_shape, config)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            output_tensor_shapes.append(output_shape)
            mcore_schedules.deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    if num_microbatches_remaining > 0:
        input_tensor = mcore_schedules.recv_forward(
            recv_shapes_for_virtual(num_warmup_microbatches), config
        )

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
            mcore_schedules.send_forward(
                output_tensor, send_shapes_for_virtual(virtual_microbatch), config
            )
            if not last_iteration:
                input_tensor = mcore_schedules.recv_forward(
                    recv_shapes_for_virtual(virtual_microbatch + 1), config
                )
        else:
            current_output_shape = send_shapes_for_virtual(virtual_microbatch)
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            output_tensor_shapes.append(current_output_shape)
            backward_output_shape = output_tensor_shapes.pop(0)
            output_tensor_grad = mcore_schedules.send_forward_recv_backward(
                output_tensor, backward_output_shape, config
            )
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
                mcore_schedules.send_backward(input_tensor_grad, backward_output_shape, config)
            else:
                input_tensor = mcore_schedules.send_backward_recv_forward(
                    input_tensor_grad, recv_shapes_for_virtual(virtual_microbatch + 1), config
                )

    if not forward_only:
        for i in range(num_warmup_microbatches):
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            backward_output_shape = output_tensor_shapes.pop(0)
            output_tensor_grad = mcore_schedules.recv_backward(backward_output_shape, config)
            input_tensor_grad = mcore_schedules.backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            mcore_schedules.send_backward(input_tensor_grad, backward_output_shape, config)

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
