from pathlib import Path


def test_hybrid_softmax_uses_shared_recompute_kv():
    source = (
        Path(__file__).parents[2]
        / 'swift'
        / 'megatron'
        / 'deltanet'
        / 'attention.py'
    ).read_text()
    hybrid = source.split('class Seq1F1BHybridSelfAttention', 1)[1]
    assert '_snapshot_kv_cache' not in hybrid
    assert 'seq1f1b_recompute_kv_init_cache' not in hybrid
    assert 'record_prefix_offset(' in hybrid
    assert 'restore_recompute_prefix(' in hybrid
    assert 'seal_recompute_kv(' in hybrid


if __name__ == '__main__':
    test_hybrid_softmax_uses_shared_recompute_kv()
