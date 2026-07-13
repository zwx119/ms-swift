# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
import time
from contextlib import contextmanager
from functools import partial

import megatron.core
import torch
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.core.utils import StragglerDetector
from megatron.training import ft_integration, get_args, get_timers, is_last_rank, pretrain, print_rank_0, training
from packaging import version
from torch.distributed.nn import all_reduce

from swift.utils import get_logger
from ...deltanet.context import get_split_progress, set_seq_split_context
from ...deltanet.rnn_sp1_model_patch import apply_rnn_sp1_model_patch
from ...deltanet.schedule_patch import (
    forward_backward_pipelining_without_interleaving_rnn_sp1,
    forward_backward_pipelining_without_interleaving_seq1f1b,
    get_seq1f1b_splits,
)
from ..patcher import patch_megatron_data_collator
from ..utils import get_batch, get_swift_datasets_provider

logger = get_logger()


class MegatronTrainer:

    def __init__(self, args):
        self.args = args
        self.stimer = StragglerDetector()
        self._patch_megatron()

    @staticmethod
    def _pad_seq1f1b_tensor(key, value, target_seq_len):
        seq_len = value.shape[1]
        if seq_len == target_seq_len:
            return value
        if seq_len > target_seq_len:
            raise ValueError(
                f'{key} sequence length {seq_len} exceeds Seq1F1B split sum '
                f'{target_seq_len}.'
            )

        pad_len = target_seq_len - seq_len
        if key == 'labels':
            pad_value = -100
            pad = value.new_full((value.shape[0], pad_len), pad_value)
        elif key == 'position_ids':
            if seq_len == 0:
                start = 0
            else:
                start = value[:, -1:].clone() + 1
            offsets = torch.arange(pad_len, device=value.device, dtype=value.dtype).unsqueeze(0)
            pad = start + offsets
        else:
            pad_value = 0
            pad = value.new_full((value.shape[0], pad_len), pad_value)
        return torch.cat((value, pad), dim=1).contiguous()

    @contextmanager
    def _get_iters(self, train_dataset, val_dataset):
        origin_initialize_megatron = training.initialize_megatron

        def initialize_megatron(*_args, **kwargs):
            res = origin_initialize_megatron(*_args, **kwargs)
            args = get_args()
            data_parallel_size = mpu.get_data_parallel_world_size()
            step_batch_size = args.micro_batch_size * data_parallel_size
            if args.train_iters is None and args.max_epochs is not None:
                if hasattr(train_dataset, '__len__'):
                    dataset_sample = len(train_dataset) // step_batch_size * step_batch_size
                    args.train_iters = dataset_sample * args.max_epochs // args.global_batch_size
                else:
                    raise ValueError(
                        'You are using a streaming training dataset. Please explicitly specify `--train_iters`.')
            if args.eval_iters < 0:
                if val_dataset is None:
                    args.eval_iters = 0
                elif hasattr(val_dataset, '__len__'):
                    dataset_sample = len(val_dataset) // step_batch_size * step_batch_size
                    args.eval_iters = max(dataset_sample // args.global_batch_size, 1)
                else:
                    raise ValueError(
                        'You are using a streaming validation dataset. Please explicitly specify `--eval_iters`.')
            return res

        training.initialize_megatron = initialize_megatron
        try:
            yield
        finally:
            training.initialize_megatron = origin_initialize_megatron

    @staticmethod
    def new_cyclic_iter(iterable):
        args = get_args()
        i = 0
        while True:
            is_training = getattr(args, 'is_training', False)
            if is_training:
                logger.info(f'The training of Epoch {i} starts...')
            if is_training and args.max_epochs and i >= args.max_epochs - 1:
                it = iter(iterable)
                num_batches = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)
                x = [next(it) for _ in range(num_batches)]
                while True:
                    try:
                        next_x = [next(it) for _ in range(num_batches)]
                    except StopIteration:
                        break
                    yield from x
                    x = next_x
                logger.info(f'Training of {i + 1} epochs has been completed, the training has finished.')
                x[0]['is_finished'] = True
                yield from x
            else:
                for x in iterable:
                    yield x
            i += 1

    @staticmethod
    @contextmanager
    def _training_context():
        args = get_args()
        args.is_training = True
        try:
            yield
        finally:
            args.is_training = False

    def _replace_data_iterator(self, data_iterator):
        return data_iterator

    @staticmethod
    def _dump_grad_debug(model, train_step_result):
        debug_dir = os.environ.get('SWIFT_DELTANET_GRAD_DEBUG_DIR')
        tensor_debug_dir = os.environ.get('SEQ1F1B_GRAD_TENSOR_DEBUG_DIR')
        if not debug_dir and not tensor_debug_dir:
            return

        args = get_args()
        iteration = int(getattr(args, 'curr_iteration', -1)) + 1
        max_tensor_iters = int(os.environ.get('SEQ1F1B_GRAD_TENSOR_DEBUG_MAX_ITERS', '0') or '0')
        dump_tensors = bool(tensor_debug_dir) and (max_tensor_iters <= 0 or iteration <= max_tensor_iters)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        record = None
        if debug_dir:
            record = {
                'iteration': iteration,
                'rank': rank,
                'pipeline_rank': mpu.get_pipeline_model_parallel_rank(),
                'tensor_rank': mpu.get_tensor_model_parallel_rank(),
                'pipe_sp_splits': int(getattr(args, 'pipe_sp_splits', 1)),
                'returned_grad_norm': None if train_step_result[5] is None else float(train_step_result[5]),
                'param_count': 0,
                'grad_count': 0,
                'numel': 0,
                'sum': 0.0,
                'abs_sum': 0.0,
                'sq_sum': 0.0,
                'max_abs': 0.0,
                'params': [],
            }
        tensor_payload = None
        if dump_tensors:
            tensor_payload = {
                'iteration': iteration,
                'rank': rank,
                'pipeline_rank': mpu.get_pipeline_model_parallel_rank(),
                'tensor_rank': mpu.get_tensor_model_parallel_rank(),
                'params': {},
            }

        for chunk_idx, model_chunk in enumerate(model):
            for name, param in model_chunk.named_parameters():
                if record is not None:
                    record['param_count'] += 1
                grad = getattr(param, 'main_grad', None)
                grad_source = 'main_grad'
                if grad is None:
                    grad = param.grad
                    grad_source = 'grad'
                if grad is None:
                    continue
                grad_f = grad.detach().float()
                full_name = f'chunk{chunk_idx}.{name}'
                if tensor_payload is not None:
                    tensor_payload['params'][full_name] = grad_f.cpu()
                if record is not None:
                    grad_sum = grad_f.sum().item()
                    grad_abs_sum = grad_f.abs().sum().item()
                    grad_sq_sum = grad_f.square().sum().item()
                    grad_max_abs = grad_f.abs().max().item() if grad_f.numel() else 0.0
                    numel = grad_f.numel()
                    record['grad_count'] += 1
                    record['numel'] += numel
                    record['sum'] += grad_sum
                    record['abs_sum'] += grad_abs_sum
                    record['sq_sum'] += grad_sq_sum
                    record['max_abs'] = max(record['max_abs'], grad_max_abs)
                    record['params'].append({
                        'name': full_name,
                        'source': grad_source,
                        'numel': numel,
                        'sum': grad_sum,
                        'abs_sum': grad_abs_sum,
                        'sq_sum': grad_sq_sum,
                        'norm': grad_sq_sum**0.5,
                        'max_abs': grad_max_abs,
                    })

        if record is not None:
            record['norm'] = record['sq_sum']**0.5
            os.makedirs(debug_dir, exist_ok=True)
            path = os.path.join(debug_dir, f'grad_stats_rank{rank}.jsonl')
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, sort_keys=True) + '\n')
        if tensor_payload is not None:
            os.makedirs(tensor_debug_dir, exist_ok=True)
            path = os.path.join(tensor_debug_dir, f'grad_tensors_rank{rank}_iter{iteration}.pt')
            torch.save(tensor_payload, path)

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
        with self._training_context():
            new_data_iterator = self._replace_data_iterator(data_iterator)
            result = self._origin_train_step(forward_step_func, new_data_iterator, model, optimizer,
                                             opt_param_scheduler, config)
            self._dump_grad_debug(model, result)
            return result

    # Code borrowed from NVIDIA/Megatron-LM
    def evaluate(self,
                 forward_step_func,
                 data_iterator,
                 model,
                 process_non_loss_data_func,
                 config,
                 verbose=False,
                 non_loss_data_func=None):
        """Evaluation."""
        args = get_args()
        timers = get_timers()

        timers('evaluate', log_level=0).start(barrier=True)

        if args.vision_pretraining and args.vision_pretraining_type == 'dino':
            from megatron.legacy.model.vision.knn_monitor import compute_feature_bank
            compute_feature_bank(model)

        # Turn on evaluation mode which disables dropout.
        for model_module in model:
            model_module.eval()

        # Disable result validation during evaluation
        rerun_state_machine = get_rerun_state_machine()
        rerun_mode = rerun_state_machine.get_mode()
        rerun_state_machine.set_mode(RerunMode.DISABLED)

        total_loss_dict = {}

        # make validation batch size independent from training batch size
        eval_batch_size = args.global_batch_size
        eval_num_microbatches = eval_batch_size // (args.micro_batch_size * args.data_parallel_size)
        megatron_core_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
        with torch.no_grad():
            iteration = 0
            if verbose:
                print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
            while iteration < args.eval_iters:
                iteration += 1
                if verbose:
                    print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

                # Use the (possibly Seq1F1B-patched) schedule so that DeltaNet
                # evaluation slices sequence chunks consistently with training.
                forward_backward_func = self.get_forward_backward_func()
                # Don't care about timing during evaluation
                config.timers = None
                ft_integration.on_eval_step_start()
                new_data_iterator = self._replace_data_iterator(data_iterator)
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=new_data_iterator,
                    model=model,
                    num_microbatches=eval_num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True)
                ft_integration.on_eval_step_end()
                config.timers = get_timers()

                # Empty unused memory
                if args.empty_unused_memory_level >= 1:
                    torch.cuda.empty_cache()

                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    if megatron_core_013:
                        for key in loss_dicts[0].keys():
                            if key not in total_loss_dict:
                                total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float).cuda()
                            val = [x[key].view(-1) for x in loss_dicts]
                            if val[0].numel() == 2:
                                val = torch.vstack(val).sum(dim=0)
                                torch.distributed.all_reduce(
                                    val, group=mpu.get_data_parallel_group(with_context_parallel=True))
                                total_loss_dict[key] += val
                            elif val[0].numel() == 1:
                                val = torch.cat(val).sum()
                                total_loss_dict[key][0] += val
                                total_loss_dict[key][1] += len(loss_dicts)
                            else:
                                raise ValueError(f'Invalid value shape: {val[0].shape} for key {key}')
                    else:
                        # Reduce across processes.
                        for loss_dict in loss_dicts:
                            for key in loss_dict:
                                if key not in total_loss_dict:
                                    total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float).cuda()
                                val = loss_dict[key]
                                if isinstance(val, tuple) or isinstance(val, list):
                                    total_loss_dict[key][0] += val[0]
                                    total_loss_dict[key][1] += val[1]
                                else:
                                    total_loss_dict[key][0] += val
                                    total_loss_dict[key][1] += 1
                args.consumed_valid_samples += eval_batch_size

                if args.exit_duration_in_mins:
                    train_time = (time.time() - training._TRAIN_START_TIME) / 60.0
                    done_cuda = torch.tensor([train_time > args.exit_duration_in_mins], dtype=torch.int, device='cuda')
                    torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
                    done = done_cuda.item()
                    if done:
                        rerun_state_machine.set_mode(rerun_mode)
                        print_rank_0('Exiting during evaluation, timelimit reached')
                        return None, None, True

            collected_non_loss_data = None
            if non_loss_data_func is not None:
                collected_non_loss_data = non_loss_data_func(model)
            elif process_non_loss_data_func is not None and is_last_rank():
                collected_non_loss_data = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=get_num_microbatches(),
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True,
                    collect_non_loss_data=True)

        # Move model back to the train mode.
        for model_module in model:
            model_module.train()

        for key in total_loss_dict:
            numerator, denominator = total_loss_dict[key]
            total_loss_dict[key] = numerator / denominator

        timers('evaluate').stop()
        timers.log(['evaluate'])

        rerun_state_machine.set_mode(rerun_mode)

        rerun_state_machine.set_mode(rerun_mode)

        return total_loss_dict, collected_non_loss_data, False

    def _patch_megatron(self):
        apply_rnn_sp1_model_patch()
        # support max_epochs
        self._origin_train_step = training.train_step
        training.train_step = self.train_step
        self._origin_get_forward_backward_func = training.get_forward_backward_func
        training.get_forward_backward_func = self.get_forward_backward_func
        training.cyclic_iter = self.new_cyclic_iter
        # patch training_log
        self._origin_training_log = training.training_log
        # patch evaluate
        self._origin_evaluate = training.evaluate
        training.evaluate = self.evaluate

    def get_forward_backward_func(self):
        args = get_args()
        if (getattr(args, 'use_deltanet', False) or getattr(args, 'use_mamba3', False)) and getattr(args, 'pipe_sp_splits', 1) > 1:
            if mpu.get_pipeline_model_parallel_world_size() <= 1:
                raise ValueError('Seq1F1B currently requires pipeline_model_parallel_size > 1.')
            if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
                raise ValueError('Seq1F1B patch currently supports non-interleaved PP only.')
            if getattr(args, 'deltanet_rnn_sp1_baseline', False):
                return forward_backward_pipelining_without_interleaving_rnn_sp1
            return forward_backward_pipelining_without_interleaving_seq1f1b
        return self._origin_get_forward_backward_func()

    # Code borrowed from NVIDIA/Megatron-LM
    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  loss_mask: torch.Tensor,
                  full_num_tokens: torch.Tensor = None,
                  num_seq_splits: int = 1):
        """Loss function.

        Args:
            output_tensor (torch.Tensor): The tensor with the losses
            loss_mask (torch.Tensor): Used to mask out some portions of the loss
            full_num_tokens (torch.Tensor): For Seq1F1B sequence splits, the
                non-padded token count of the *full* (unsliced) sequence.
            num_seq_splits (int): Seq1F1B ``pipe_sp_splits`` of this run.

        Returns:
            the loss scalar for this micro-batch
            the number of non-padded tokens in this microbatch
            a dict containing reporting metrics on the loss and number of tokens across
                the data parallel ranks
        """
        args = get_args()

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

        megatron_core_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
        if args.context_parallel_size > 1 and not megatron_core_013:
            loss = all_reduce(loss, group=mpu.get_context_parallel_group())

        # Check individual rank losses are not NaN prior to DP all-reduce.
        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isnan,
                message='found NaN in local forward loss calculation',
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isinf,
                message='found Inf in local forward loss calculation',
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
        # Check for spiky loss
        if args.check_for_spiky_loss:
            # define spiky loss as a loss that's 10x the max loss observed
            SPIKY_LOSS_FACTOR = 10
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=SPIKY_LOSS_FACTOR,
                    context='loss',
                ),
                message='Spiky loss',
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=False,
            )
        # Reduce loss for logging.
        reporting_loss = loss.clone().detach()
        lm_loss = loss[0]
        if not megatron_core_013:
            # fix megatron-lm bug
            # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
            lm_loss = lm_loss / mpu.get_context_parallel_world_size()
            reporting_loss = (reporting_loss[0], reporting_loss[1])
        else:
            lm_loss = lm_loss.clone()
        local_num_tokens = loss[1].clone().detach().to(torch.int)
        if (num_seq_splits > 1 and full_num_tokens is not None and not getattr(args, 'calculate_per_token_loss', False)):
            # Seq1F1B: Megatron's forward_step divides each chunk's loss sum by
            # its *local* token count and by num_microbatches * num_seq_splits.
            # To make the summed gradient identical to the unsplit baseline
            # (chunk_sum / full_tokens / num_microbatches), scale the loss by
            # the split count and report the full-sequence token count instead.
            # `reporting_loss` above keeps the true per-chunk statistics.
            lm_loss = lm_loss * num_seq_splits
            local_num_tokens = full_num_tokens.clone().detach().to(torch.int)
        return (
            lm_loss,
            local_num_tokens,
            {
                'lm loss': reporting_loss
            },
        )

    def loss_func_group(self, output_tensors, *, loss_masks):
        """Loss for the DeltaNet RNN-SP1 grouped naive baseline."""
        args = get_args()
        loss_sum = None
        total_tokens = None
        for output_tensor, loss_mask in zip(output_tensors, loss_masks):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            chunk_sum = torch.sum(losses.view(-1) * loss_mask)
            chunk_tokens = loss_mask.sum()
            loss_sum = chunk_sum if loss_sum is None else loss_sum + chunk_sum
            total_tokens = chunk_tokens if total_tokens is None else total_tokens + chunk_tokens

        loss = torch.cat([loss_sum.view(1), total_tokens.view(1)])
        megatron_core_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
        if args.context_parallel_size > 1 and not megatron_core_013:
            loss = all_reduce(loss, group=mpu.get_context_parallel_group())

        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isnan,
                message='found NaN in local forward loss calculation',
                tolerance=0.0,
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isinf,
                message='found Inf in local forward loss calculation',
                tolerance=0.0,
                fatal=True,
            )

        reporting_loss = loss.clone().detach()
        lm_loss = loss[0]
        if not megatron_core_013:
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
            lm_loss = lm_loss / mpu.get_context_parallel_world_size()
            reporting_loss = (reporting_loss[0], reporting_loss[1])
        else:
            lm_loss = lm_loss.clone()

        return (
            lm_loss,
            loss[1].clone().detach().to(torch.int),
            {
                'lm loss': reporting_loss
            },
        )

    def forward_step(self, data_iterator, model):
        timers = get_timers()
        args = get_args()
        use_deltanet_sp = (getattr(args, 'use_deltanet', False) or getattr(args, 'use_mamba3', False)) and getattr(args, 'pipe_sp_splits', 1) > 1
        use_rnn_sp1_baseline = use_deltanet_sp and getattr(args, 'deltanet_rnn_sp1_baseline', False)
        progress = get_split_progress()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            if (
                use_deltanet_sp
                and not use_rnn_sp1_baseline
                and progress['cached_batch'] is not None
                and progress['idx'] > 0
            ):
                data = progress['cached_batch']
            else:
                data = get_batch(data_iterator)
                if use_deltanet_sp and not use_rnn_sp1_baseline:
                    progress['cached_batch'] = data
        timers('batch-generator').stop()

        full_num_tokens = None
        if use_rnn_sp1_baseline:
            data, loss_masks = self._prepare_deltanet_rnn_sp1_batch(data)
        elif getattr(args, 'use_deltanet', False) or getattr(args, 'use_mamba3', False):
            data, full_num_tokens = self._prepare_deltanet_batch(data)
        else:
            set_seq_split_context()

        with self.stimer:
            output_tensor = model(**data)
        if use_rnn_sp1_baseline:
            return output_tensor, partial(self.loss_func_group, loss_masks=loss_masks)
        labels = data.get('labels')
        loss_mask = None if labels is None else (labels != -100).float()
        return output_tensor, partial(
            self.loss_func,
            loss_mask=loss_mask,
            full_num_tokens=full_num_tokens,
            num_seq_splits=getattr(args, 'pipe_sp_splits', 1) if use_deltanet_sp else 1,
        )

    def _prepare_deltanet_rnn_sp1_batch(self, data):
        args = get_args()
        pipe_sp_splits = getattr(args, 'pipe_sp_splits', 1)
        if pipe_sp_splits <= 1:
            raise ValueError('`deltanet_rnn_sp1_baseline` requires pipe_sp_splits > 1.')

        grouped = {}
        loss_masks = None
        for key, value in data.items():
            if key in {'input_ids', 'labels', 'position_ids'} and value is not None:
                seq_len = value.shape[1]
                if seq_len % pipe_sp_splits != 0:
                    raise ValueError(f'{key} sequence length {seq_len} is not divisible by {pipe_sp_splits}.')
                chunk = seq_len // pipe_sp_splits
                chunks = [
                    value[:, idx * chunk:(idx + 1) * chunk].contiguous()
                    for idx in range(pipe_sp_splits)
                ]
                grouped[key] = chunks
                if key == 'labels':
                    loss_masks = [(part != -100).float() for part in chunks]
            elif key == 'attention_mask':
                grouped[key] = None
            elif key == 'packed_seq_params':
                grouped[key] = None
            else:
                grouped[key] = value

        if loss_masks is None and mpu.is_pipeline_last_stage():
            raise ValueError('DeltaNet RNN-SP1 baseline expects labels in the training batch.')
        set_seq_split_context(0, 1)
        progress = get_split_progress()
        progress['idx'] = 0
        progress['cached_batch'] = None
        progress['microbatch_key'] = 0
        return grouped, loss_masks

    def _prepare_deltanet_batch(self, data):
        args = get_args()
        pipe_sp_splits = getattr(args, 'pipe_sp_splits', 1)
        if pipe_sp_splits <= 1:
            set_seq_split_context(0, 1)
            data = dict(data)
            data['attention_mask'] = None
            data['packed_seq_params'] = None
            return data, None

        progress = get_split_progress()
        micro_sp_idx = progress['idx']
        microbatch_key = int(progress.get('microbatch_key', 0))
        start = None
        end = None
        full_num_tokens = None
        sliced = {}
        split_lengths = None
        for key, value in data.items():
            if key in {'input_ids', 'labels', 'position_ids'} and value is not None:
                seq_len = value.shape[1]
                if split_lengths is None:
                    split_lengths = get_seq1f1b_splits(args)
                target_seq_len = sum(split_lengths)
                if seq_len != target_seq_len:
                    value = self._pad_seq1f1b_tensor(key, value, target_seq_len)
                    seq_len = value.shape[1]
                if seq_len != target_seq_len:
                    raise ValueError(
                        f'{key} sequence length {seq_len} does not match previous '
                        f'Seq1F1B split sum {target_seq_len}.'
                    )
                start = sum(split_lengths[:micro_sp_idx])
                end = start + split_lengths[micro_sp_idx]
                if key == 'labels':
                    full_num_tokens = (value != -100).sum()
                sliced[key] = value[:, start:end].contiguous()
            elif key == 'attention_mask':
                sliced[key] = None
            elif key == 'packed_seq_params':
                sliced[key] = None
            else:
                sliced[key] = value

        set_seq_split_context(micro_sp_idx, pipe_sp_splits, start, end, microbatch_key=microbatch_key)
        progress['idx'] = (micro_sp_idx + 1) % pipe_sp_splits
        if progress['idx'] == 0:
            progress['cached_batch'] = None
            progress['microbatch_key'] = microbatch_key + 1
        return sliced, full_num_tokens

    def train(self, train_dataset, val_dataset, data_collator):
        args = self.args
        datasets_provider = get_swift_datasets_provider(train_dataset, val_dataset)
        datasets_provider.is_distributed = True
        with patch_megatron_data_collator(data_collator), self._get_iters(train_dataset, val_dataset):
            extra_args_provider = args.megatron_model_meta.extra_args_provider
            pretrain(
                datasets_provider,
                args.megatron_model_meta.model_provider,
                ModelType.encoder_or_decoder,
                self.forward_step,
                extra_args_provider=extra_args_provider,
                args_defaults=args.extra_args)
