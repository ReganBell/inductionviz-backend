import torch

from backend.app import (
    DEVICE,
    VOCAB,
    analyze_text,
    encode,
    logits_zero_attn_path,
    make_attn_only,
    sample_from_logits,
    sample_generate_logits_and_attn,
)


def test_zero_attn_path_matches_vocab_shape():
    model = make_attn_only(VOCAB, n_layers=1, d_model=64, n_heads=4, n_ctx=32)
    tokens = torch.randint(0, VOCAB, (2, 5), device=DEVICE)
    logits = logits_zero_attn_path(model, tokens)
    assert logits.shape == (2, VOCAB)


def test_sample_generate_collects_attention_shapes():
    text = "the quick brown fox jumps over"
    seed_ids = encode(text)
    steps = 3
    ctx = len(seed_ids) + steps
    model = make_attn_only(VOCAB, n_layers=1, d_model=64, n_heads=4, n_ctx=ctx)

    seed_tensor = torch.tensor(seed_ids, device=DEVICE, dtype=torch.long).unsqueeze(0)

    gen_ids, logits, attn, loss = sample_generate_logits_and_attn(model, seed_tensor, steps)

    assert len(gen_ids) == steps
    assert logits.shape == (steps, VOCAB)
    assert len(attn) == steps
    assert loss >= 0

    first_layer = attn[0][0]
    assert first_layer.shape[-1] == seed_tensor.shape[1]
    assert first_layer.shape[-2] == seed_tensor.shape[1]

    last_layer = attn[-1][0]
    expected_length = seed_tensor.shape[1] + steps - 1
    assert last_layer.shape[-1] == expected_length
    assert last_layer.shape[-2] == expected_length


def test_analyze_text_returns_positions():
    models = {
        "t1": make_attn_only(VOCAB, n_layers=1, d_model=64, n_heads=4, n_ctx=64),
        "t2": make_attn_only(VOCAB, n_layers=1, d_model=64, n_heads=4, n_ctx=64),
    }
    result = analyze_text("hello world",
                          top_k=5,
                          models=models)
    assert len(result["tokens"]) == 2
    assert len(result["positions"]) == 1
    pos = result["positions"][0]
    assert set(pos["topk"].keys()) == {"bigram", "t1", "t2"}


@torch.no_grad()
def test_sample_respects_top_k():
    logits = torch.arange(0, 10, dtype=torch.float32)
    allowed = set(range(5, 10))
    torch.manual_seed(0)
    draws = {sample_from_logits(logits, top_k=5, temperature=0.7) for _ in range(100)}
    assert draws.issubset(allowed)


@torch.no_grad()
def test_sample_requires_positive_temperature():
    logits = torch.zeros(5)
    try:
        sample_from_logits(logits, temperature=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive temperature")
