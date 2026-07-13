from swift.megatron.mamba3.spec import get_mamba3_gpt_layer_spec


def test_hybrid_block_keeps_final_layernorm():
    block = get_mamba3_gpt_layer_spec(
        use_transformer_engine=True,
        normalization="LayerNorm",
        num_layers=4,
        hybrid_attention_period=4,
        hybrid_attention_offset=0,
    )

    assert block.layer_norm is not None


if __name__ == "__main__":
    test_hybrid_block_keeps_final_layernorm()
    print("SWIFT_MAMBA3_HYBRID_FINAL_LAYERNORM_OK")
