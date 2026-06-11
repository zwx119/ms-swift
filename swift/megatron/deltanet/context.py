# Copyright (c) Alibaba, Inc. and its affiliates.

"""Process-local Seq1F1B state used by DeltaNet attention.

Megatron-Core transformer layers do not pass arbitrary per-microbatch metadata
down to the attention module. Swift drives one process per rank, so a small
process-local context is the least invasive way to expose the current sequence
split to every DeltaNet layer without forking the whole MCore layer stack.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SeqSplitContext:
    """Current sequence-split metadata for one forward call."""

    micro_sp_idx: int = 0
    pipe_sp_splits: int = 1
    start: Optional[int] = None
    end: Optional[int] = None


_CURRENT_CONTEXT = SeqSplitContext()


def set_seq_split_context(
    micro_sp_idx: int = 0,
    pipe_sp_splits: int = 1,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> None:
    """Publish the active split before calling the model."""

    global _CURRENT_CONTEXT
    _CURRENT_CONTEXT = SeqSplitContext(
        micro_sp_idx=int(micro_sp_idx),
        pipe_sp_splits=int(pipe_sp_splits),
        start=start,
        end=end,
    )


def get_seq_split_context() -> SeqSplitContext:
    """Return the active split context."""

    return _CURRENT_CONTEXT


def reset_seq_split_context() -> None:
    """Reset to the unsplit default."""

    set_seq_split_context()


# Seq1F1B split progress shared between the schedule patch and the trainer.
#
# The schedule decides the global forward order (warmup + 1F1B), while the
# trainer's ``forward_step`` decides which sequence chunk to feed. Keeping the
# chunk cursor here (and resetting it at the start of every forward-backward
# run) guarantees the two can never drift apart, e.g. across evaluation runs,
# Megatron rerun-state-machine replays, or an aborted iteration.
_SPLIT_PROGRESS = {
    'idx': 0,
    'cached_batch': None,
}


def get_split_progress() -> dict:
    """Return the mutable per-process split progress."""

    return _SPLIT_PROGRESS


def reset_split_progress() -> None:
    """Reset the chunk cursor before a new forward-backward run."""

    _SPLIT_PROGRESS['idx'] = 0
    _SPLIT_PROGRESS['cached_batch'] = None
    reset_seq_split_context()

