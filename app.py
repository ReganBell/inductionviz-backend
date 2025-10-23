from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional
import os
from pathlib import Path
from collections import defaultdict
import psutil
import logging

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import tiktoken

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")
VOCAB = enc.n_vocab


def log_memory_usage(context: str = ""):
    """Log current memory usage in MB"""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    logger.info(f"Memory usage {context}: {mem_mb:.2f} MB")
    return mem_mb


def encode(text: str) -> List[int]:
    return enc.encode(text)


def decode(ids: List[int]) -> str:
    return enc.decode(ids)


def _corpus_path() -> Path:
    override = os.environ.get("BIGRAM_CORPUS_PATH")
    return Path(override) if override else Path(__file__).resolve().parent.parent / "words.txt"


def _load_corpus_bigram_counts() -> Dict[int, Dict[int, int]]:
    with open('words.txt', 'r') as f:
        content = f.read()
        counts = defaultdict(lambda: defaultdict(int))
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            ids = encode(line)
            for a, b in zip(ids[:-1], ids[1:]):
                counts[a][b] += 1
    return {prev: dict(next_counts) for prev, next_counts in counts.items()}


# Disabled to reduce memory usage for Railway deployment
# CORPUS_BIGRAM_COUNTS: Dict[int, Dict[int, int]] = _load_corpus_bigram_counts()
CORPUS_BIGRAM_COUNTS: Dict[int, Dict[int, int]] = {}

log_memory_usage("before loading models")

MODELS: Dict[str, HookedTransformer] = {
    "t1": HookedTransformer.from_pretrained("attn-only-1l", device=DEVICE).to(DEVICE).eval(),
    "t2": HookedTransformer.from_pretrained("attn-only-2l", device=DEVICE).to(DEVICE).eval(),
}

log_memory_usage("after loading both models")


def make_attn_only(
    vocab_size: int,
    n_layers: int,
    d_model: int = 256,
    n_heads: int = 4,
    n_ctx: int = 256,
) -> HookedTransformer:
    from transformer_lens import HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_model // n_heads,
        n_ctx=n_ctx,
        d_vocab=vocab_size,
        attn_only=True,
        use_attn_result=True,
        device=DEVICE,
    )
    return HookedTransformer(cfg).to(DEVICE)


@torch.no_grad()
def logits_zero_attn_path(model: HookedTransformer, tokens: torch.Tensor) -> torch.Tensor:
    E = model.W_E
    P = model.W_pos
    positions = torch.arange(tokens.size(1), device=tokens.device)
    x = E[tokens] + P[positions]
    x_last = x[:, -1, :]
    x_last = model.ln_final(x_last)
    logits = x_last @ model.W_U
    return logits


@torch.no_grad()
def calculate_composition_scores(model: HookedTransformer, subtract_baseline: bool = True) -> Dict[str, List[List[float]]]:
    """
    Calculate Q-, K-, and V-composition scores between attention heads.

    For a 2-layer model, this measures how much layer 1 heads compose with layer 0 heads.

    Q-Composition: ||W_QK_h2^T @ W_OV_h1||_F / (||W_QK_h2^T||_F * ||W_OV_h1||_F)
    K-Composition: ||W_QK_h2 @ W_OV_h1||_F / (||W_QK_h2||_F * ||W_OV_h1||_F)
    V-Composition: ||W_OV_h2 @ W_OV_h1||_F / (||W_OV_h2||_F * ||W_OV_h1||_F)

    Where:
    - W_QK = W_Q @ W_K^T (query-key circuit)
    - W_OV = W_V @ W_O (value-output circuit)

    By default, subtracts empirical baseline from random matrices (0.0442 for d_model=512).
    This helps identify significant composition patterns above chance.
    """
    if model.cfg.n_layers < 2:
        # Need at least 2 layers for composition
        return {"q_composition": [], "k_composition": [], "v_composition": []}

    n_heads = model.cfg.n_heads

    # Only compute composition from layer 0 -> layer 1
    layer0 = 0
    layer1 = 1

    # Get weight matrices for layer 0
    # W_Q, W_K, W_V: [n_heads, d_model, d_head]
    # W_O: [n_heads, d_head, d_model]
    W_Q_0 = model.W_Q[layer0]  # [n_heads, d_model, d_head]
    W_K_0 = model.W_K[layer0]  # [n_heads, d_model, d_head]
    W_V_0 = model.W_V[layer0]  # [n_heads, d_model, d_head]
    W_O_0 = model.W_O[layer0]  # [n_heads, d_head, d_model]

    # Get weight matrices for layer 1
    W_Q_1 = model.W_Q[layer1]
    W_K_1 = model.W_K[layer1]
    W_V_1 = model.W_V[layer1]
    W_O_1 = model.W_O[layer1]

    # Compute W_OV and W_QK for each head
    # W_OV = W_V @ W_O for each head
    W_OV_0 = torch.einsum('hdi,hio->hdo', W_V_0, W_O_0)  # [n_heads, d_model, d_model]
    W_OV_1 = torch.einsum('hdi,hio->hdo', W_V_1, W_O_1)  # [n_heads, d_model, d_model]

    # W_QK = W_Q @ W_K^T for each head
    W_QK_0 = torch.einsum('hqi,hki->hqk', W_Q_0, W_K_0)  # [n_heads, d_model, d_model]
    W_QK_1 = torch.einsum('hqi,hki->hqk', W_Q_1, W_K_1)  # [n_heads, d_model, d_model]

    # Initialize score matrices: [layer1_head, layer0_head]
    q_scores = torch.zeros(n_heads, n_heads)
    k_scores = torch.zeros(n_heads, n_heads)
    v_scores = torch.zeros(n_heads, n_heads)

    for h2 in range(n_heads):  # layer 1 heads
        for h1 in range(n_heads):  # layer 0 heads
            # Q-Composition: W_QK_h2^T @ W_OV_h1
            q_product = W_QK_1[h2].T @ W_OV_0[h1]  # [d_model, d_model]
            q_norm_product = torch.norm(q_product, p='fro')
            q_norm_qk = torch.norm(W_QK_1[h2].T, p='fro')
            q_norm_ov = torch.norm(W_OV_0[h1], p='fro')
            q_scores[h2, h1] = q_norm_product / (q_norm_qk * q_norm_ov + 1e-8)

            # K-Composition: W_QK_h2 @ W_OV_h1
            k_product = W_QK_1[h2] @ W_OV_0[h1]  # [d_model, d_model]
            k_norm_product = torch.norm(k_product, p='fro')
            k_norm_qk = torch.norm(W_QK_1[h2], p='fro')
            k_norm_ov = torch.norm(W_OV_0[h1], p='fro')
            k_scores[h2, h1] = k_norm_product / (k_norm_qk * k_norm_ov + 1e-8)

            # V-Composition: W_OV_h2 @ W_OV_h1
            v_product = W_OV_1[h2] @ W_OV_0[h1]  # [d_model, d_model]
            v_norm_product = torch.norm(v_product, p='fro')
            v_norm_ov2 = torch.norm(W_OV_1[h2], p='fro')
            v_norm_ov1 = torch.norm(W_OV_0[h1], p='fro')
            v_scores[h2, h1] = v_norm_product / (v_norm_ov2 * v_norm_ov1 + 1e-8)

    # Subtract empirical baseline for random matrices
    # This baseline was computed empirically for d_model=512 matrices
    # E[||A @ B||_F / (||A||_F * ||B||_F)] ≈ 0.0442 for random Gaussian matrices
    if subtract_baseline:
        baseline = 0.0442
        q_scores = q_scores - baseline
        k_scores = k_scores - baseline
        v_scores = v_scores - baseline

    # Convert to lists for JSON serialization
    # Each matrix is [layer1_head, layer0_head]
    return {
        "q_composition": q_scores.cpu().tolist(),
        "k_composition": k_scores.cpu().tolist(),
        "v_composition": v_scores.cpu().tolist(),
    }


def _resolve_top_k(top_k: int, vocab_size: int) -> int:
    if top_k <= 0 or top_k > vocab_size:
        return vocab_size
    return top_k


def _build_bigram_logit_map(ids: List[int], vocab: int, laplace: float = 1.0) -> Dict[int, Optional[torch.Tensor]]:
    logit_map: Dict[int, Optional[torch.Tensor]] = {}
    if CORPUS_BIGRAM_COUNTS:
        unique_prev = {int(pid) for pid in ids[:-1]}
        print('unique_prev', unique_prev)
        for prev_id in unique_prev:
            next_counts = CORPUS_BIGRAM_COUNTS.get(prev_id)
            if not next_counts:
                logit_map[prev_id] = None  # Explicitly mark as missing
                continue
            vec = torch.full((vocab,), laplace, dtype=torch.float32, device=DEVICE)
            for next_id, count in next_counts.items():
                vec[next_id] += float(count)
            probs = vec / vec.sum()
            logit_map[prev_id] = torch.log(probs)
        # Ensure all unique_prev tokens are in the map (either with logits or None)
        for prev_id in unique_prev:
            if prev_id not in logit_map:
                logit_map[prev_id] = None
    return logit_map


@torch.no_grad()
def _run_with_cache(model: HookedTransformer, toks: torch.Tensor):
    # Capture both attention patterns and value vectors
    pattern_names = [f"blocks.{layer}.attn.hook_pattern" for layer in range(model.cfg.n_layers)]
    value_names = [f"blocks.{layer}.attn.hook_v" for layer in range(model.cfg.n_layers)]
    names = pattern_names + value_names

    logits, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: n in names,
        return_type="logits",
    )
    return logits.squeeze(0), cache


@torch.no_grad()
def _calculate_all_head_deltas(model: HookedTransformer, toks: torch.Tensor, position: int, next_token_id: int, context_token_ids: List[int], top_k: int = 10):
    """
    Calculate delta logits for ablating each head at a given position.
    Only considers deltas for tokens that actually appear in the context.
    Returns dict with rich information about what each head is doing.
    """
    # Get baseline logits at this position
    logits_base = model(toks)[0, position]  # [vocab]

    # Get unique token IDs from context (tokens the head could be attending to)
    context_vocab = set(context_token_ids)

    deltas = {}

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            # Create hook to zero this specific head
            # Need to use a closure to capture the current head value
            def make_zero_head_hook(head_idx):
                def zero_head_hook(v, hook):
                    v[:, :, head_idx, :] = 0.0
                    return v
                return zero_head_hook

            # Run model with head ablated
            # Use hook_z (before W_O) not hook_result (after W_O)
            logits_ablated = model.run_with_hooks(
                toks,
                fwd_hooks=[(f"blocks.{layer}.attn.hook_z", make_zero_head_hook(head))]
            )[0, position]

            # Calculate delta logits: baseline - ablated = head's contribution
            delta_logits = logits_base - logits_ablated  # [vocab]

            # Filter to only context tokens
            context_deltas = {tid: float(delta_logits[tid]) for tid in context_vocab}

            # Get magnitude (sum of absolute deltas for context tokens only)
            magnitude = sum(abs(d) for d in context_deltas.values())

            # Sort context tokens by delta
            sorted_context = sorted(context_deltas.items(), key=lambda x: x[1], reverse=True)

            # Get top promoted tokens from context
            top_promoted = [
                {"token": decode([int(tid)]), "id": int(tid), "delta": float(delta)}
                for tid, delta in sorted_context[:top_k]
                if delta > 0
            ]

            # Get top suppressed tokens from context
            top_suppressed = [
                {"token": decode([int(tid)]), "id": int(tid), "delta": float(delta)}
                for tid, delta in sorted_context[-top_k:][::-1]
                if delta < 0
            ]

            # Delta for the actual next token
            actual_token_delta = float(delta_logits[next_token_id])

            deltas[f"L{layer}H{head}"] = {
                "magnitude": magnitude,
                "actual_token_delta": actual_token_delta,
                "top_promoted": top_promoted,
                "top_suppressed": top_suppressed,
            }

    return deltas


@torch.no_grad()
def _calculate_value_weighted_attention(cache: Dict[str, torch.Tensor], n_layers: int, t: int) -> List[List[List[float]]]:
    """
    Calculate value-weighted attention patterns.
    For each layer and head, compute attention weights scaled by the norm of value vectors.

    value_weighted(dest, src) = attention(dest, src) * ||v_src||

    This shows "how much information is actually being moved" not just "where is attention paid"
    """
    value_weighted_attn = []

    for layer in range(n_layers):
        layer_patterns = []

        # Get attention patterns for this layer
        pattern_key = f"blocks.{layer}.attn.hook_pattern"
        value_key = f"blocks.{layer}.attn.hook_v"

        if pattern_key not in cache or value_key not in cache:
            continue

        attn_pattern = cache[pattern_key][0, :, t, :t+1]  # [n_heads, t+1] - attention from position t to all previous
        value_vectors = cache[value_key][0, :t+1, :, :]     # [t+1, n_heads, d_head]

        n_heads = attn_pattern.size(0)

        for head in range(n_heads):
            # Get attention weights for this head from position t
            attn_weights = attn_pattern[head]  # [t+1]

            # Get value vectors for this head (source positions)
            head_values = value_vectors[:, head, :]  # [t+1, d_head]

            # Calculate norms of value vectors
            value_norms = torch.norm(head_values, dim=-1)  # [t+1]

            # Calculate value-weighted attention: attention * ||value||
            value_weighted = attn_weights * value_norms  # [t+1]

            layer_patterns.append(value_weighted.detach().cpu().tolist())

        value_weighted_attn.append(layer_patterns)

    return value_weighted_attn


def _skip_trigram_positions(ids: List[int]) -> List[int]:
    positions: List[int] = []
    for t in range(2, len(ids)):
        a, b = ids[t - 2], ids[t - 1]
        for idx in range(0, t - 1):
            if ids[idx] == a and idx + 1 < t - 1 and ids[idx + 1] == b:
                positions.append(t - 1)
                break
    return sorted(set(positions))


def _topk_pack(vec: torch.Tensor, topk: int) -> List[Dict[str, float]]:
    probs = F.softmax(vec, dim=-1)
    k = min(topk, vec.size(-1))
    tv, ti = torch.topk(vec, k)
    return [
        {
            "token": decode([int(i)]),
            "id": int(i),
            "logit": float(val),
            "prob": float(probs[int(i)]),
        }
        for val, i in zip(tv.tolist(), ti.tolist())
    ]


def analyze_text(text: str, *, top_k: int = 10, compute_ablations: bool = False, models: Optional[Dict[str, HookedTransformer]] = None):
    if models is None:
        models = MODELS
    # load text from hp.txt
    # with open('hp.txt', 'r') as f:
    #     text = f.read()
    ids = encode(text)
    if len(ids) < 2:
        raise ValueError("Need at least two tokens")
    toks = torch.tensor(ids, device=DEVICE, dtype=torch.long).unsqueeze(0)
    T = toks.size(1)

    bigram_logit_map = _build_bigram_logit_map(ids, VOCAB)

    logits1_full, cache1 = _run_with_cache(models["t1"], toks)
    logits2_full, cache2 = _run_with_cache(models["t2"], toks)

    skip_positions = set(_skip_trigram_positions(ids))

    tokens_info = [{"id": int(i), "text": decode([int(i)])} for i in ids]
    positions = []

    for t in range(1, T):
        prev_id = ids[t - 1]
        next_id = ids[t]

        bigram_logits = bigram_logit_map.get(prev_id)
        bigram_available = bigram_logits is not None
        l1 = logits1_full[t - 1]
        l2 = logits2_full[t - 1]

        attn1 = [[[] for _ in range(models["t1"].cfg.n_heads)] for _ in range(models["t1"].cfg.n_layers)]
        attn2 = [[[] for _ in range(models["t2"].cfg.n_heads)] for _ in range(models["t2"].cfg.n_layers)]

        for layer in range(models["t1"].cfg.n_layers):
            for head in range(models["t1"].cfg.n_heads):
                attn1[layer][head] = cache1[f"blocks.{layer}.attn.hook_pattern"][0, head, t, :].detach().cpu().tolist()
        for layer in range(models["t2"].cfg.n_layers):
            for head in range(models["t2"].cfg.n_heads):
                attn2[layer][head] = cache2[f"blocks.{layer}.attn.hook_pattern"][0, head, t, :].detach().cpu().tolist()


        # Calculate value-weighted attention patterns
        value_weighted_attn1 = _calculate_value_weighted_attention(cache1, models["t1"].cfg.n_layers, t)
        value_weighted_attn2 = _calculate_value_weighted_attention(cache2, models["t2"].cfg.n_layers, t)

        # Calculate head deltas for this position (SLOW - only if requested)
        if compute_ablations:
            # Pass context tokens (all tokens up to and including current position)
            context_token_ids = ids[:t]  # tokens from 0 to t-1 (what the model can attend to)
            head_deltas_t1 = _calculate_all_head_deltas(models["t1"], toks, t - 1, next_id, context_token_ids)
            head_deltas_t2 = _calculate_all_head_deltas(models["t2"], toks, t - 1, next_id, context_token_ids)
        else:
            head_deltas_t1 = {}
            head_deltas_t2 = {}

        match_index = None
        for idx in range(t - 1, -1, -1):
            if ids[idx] == next_id:
                match_index = idx
                break

        attn_match = None
        if match_index is not None:
            col = match_index
            sum_t1 = float(sum(layer[:, col].sum() for layer in [
                torch.tensor(layer_heads) for layer_heads in attn1
            ]))
            sum_t2 = float(sum(layer[:, col].sum() for layer in [
                torch.tensor(layer_heads) for layer_heads in attn2
            ]))
            attn_match = {"t1": sum_t1, "t2": sum_t2}

        # Calculate losses - handle missing bigram data
        loss_bigram = float(-bigram_logits[next_id].item()) if bigram_available else None
        loss_t1 = float(-F.log_softmax(l1, dim=-1)[next_id].item())
        loss_t2 = float(-F.log_softmax(l2, dim=-1)[next_id].item())

        # Prepare topk data - handle missing bigram data
        topk_data = {
            "t1": _topk_pack(l1, top_k),
            "t2": _topk_pack(l2, top_k),
        }
        if bigram_available:
            topk_data["bigram"] = _topk_pack(bigram_logits, top_k)
        else:
            topk_data["bigram"] = None

        positions.append(
            {
                "t": t,
                "context_token": tokens_info[t - 1],
                "next_token": tokens_info[t],
                "topk": topk_data,
                "attn": {
                    "t1": attn1,
                    "t2": attn2,
                },
                "value_weighted_attn": {
                    "t1": value_weighted_attn1,
                    "t2": value_weighted_attn2,
                },
                "head_deltas": {
                    "t1": head_deltas_t1,
                    "t2": head_deltas_t2,
                },
                "losses": {
                    "bigram": loss_bigram,
                    "t1": loss_t1,
                    "t2": loss_t2,
                },
                "bigram_available": bigram_available,
                "match_index": match_index,
                "match_attention": attn_match,
                "skip_trigram": t - 1 in skip_positions,
            }
        )

    print('config', models["t1"].cfg);

    return {
        "tokens": tokens_info,
        "positions": positions,
        "device": DEVICE,
        "t1_layers": models["t1"].cfg.n_layers,
        "t1_heads": models["t1"].cfg.n_heads,
        "t2_layers": models["t2"].cfg.n_layers,
        "t2_heads": models["t2"].cfg.n_heads,
    }


class AnalyzeReq(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    top_k: int = 10
    compute_ablations: bool = False


class AnalyzeResp(BaseModel):
    tokens: List[Dict[str, object]]
    positions: List[Dict[str, object]]
    device: str
    t1_layers: int
    t1_heads: int
    t2_layers: int
    t2_heads: int


class AblateHeadReq(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    position: int
    model_name: str  # "t1" or "t2"
    layer: int
    head: int
    top_k: int = 10


class AblateHeadResp(BaseModel):
    with_head: List[Dict[str, object]]
    without_head: List[Dict[str, object]]
    delta_positive: List[Dict[str, object]]
    delta_negative: List[Dict[str, object]]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@torch.no_grad()
def ablate_head_analysis(
    text: str,
    position: int,
    model_name: str,
    layer: int,
    head: int,
    top_k: int = 10,
    models: Optional[Dict[str, HookedTransformer]] = None,
):
    if models is None:
        models = MODELS

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    model = models[model_name]
    ids = encode(text)

    # Position here is the token index we're hovering/locked on
    # The frontend sends us the index in the token array (which includes BOS at position 0)
    # For the analysis positions array, position t predicts token at index t+1
    # So we need to look at logits from position t-1 to see predictions for token at position t

    if position < 1 or position >= len(ids):
        raise ValueError(f"Position {position} out of range for text with {len(ids)} tokens")

    if layer < 0 or layer >= model.cfg.n_layers:
        raise ValueError(f"Layer {layer} out of range for model with {model.cfg.n_layers} layers")

    if head < 0 or head >= model.cfg.n_heads:
        raise ValueError(f"Head {head} out of range for model with {model.cfg.n_heads} heads")

    toks = torch.tensor(ids, device=DEVICE, dtype=torch.long).unsqueeze(0)

    # We want to see how the head affects prediction of the token at 'position'
    # So we look at logits from position-1 (which predict position)
    logit_pos = position - 1

    # Get context token IDs (tokens that the model can attend to at this position)
    context_token_ids = ids[:position]
    context_vocab = set(context_token_ids)

    # 1) Baseline logits at the position of interest
    logits_base = model(toks)[0, logit_pos]  # [vocab]

    # 2) Re-run with the chosen head zeroed
    def zero_one_head_hook(v, hook):
        # v: [batch, seq, n_heads, d_head] at hook_z (before W_O)
        v[:, :, head, :] = 0.0
        return v

    logits_no_head = model.run_with_hooks(
        toks,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", zero_one_head_hook)]
    )[0, logit_pos]  # [vocab]

    delta_logits = logits_base - logits_no_head  # the head's exact contribution to each vocab logit

    # Filter delta_logits to only context tokens
    context_deltas = {tid: float(delta_logits[tid]) for tid in context_vocab}

    # Sort context tokens by delta
    sorted_context_pos = sorted(context_deltas.items(), key=lambda x: x[1], reverse=True)
    sorted_context_neg = sorted(context_deltas.items(), key=lambda x: x[1])

    # Get top-k results - still show full vocab for with/without
    topk_with = _topk_pack(logits_base, top_k)
    topk_without = _topk_pack(logits_no_head, top_k)

    # For deltas, only show context tokens
    topk_delta = [
        {
            "token": decode([int(tid)]),
            "id": int(tid),
            "logit": float(delta),
            "prob": 0.0,  # prob doesn't make sense for deltas
        }
        for tid, delta in sorted_context_pos[:top_k]
        if delta > 0
    ]

    topk_delta_neg = [
        {
            "token": decode([int(tid)]),
            "id": int(tid),
            "logit": float(delta),
            "prob": 0.0,  # prob doesn't make sense for deltas
        }
        for tid, delta in sorted_context_neg[:top_k]
        if delta < 0
    ]

    return {
        "with_head": topk_with,
        "without_head": topk_without,
        "delta_positive": topk_delta,
        "delta_negative": topk_delta_neg,
    }


@app.post("/api/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    log_memory_usage("at start of /api/analyze request")
    try:
        result = analyze_text(req.text, top_k=req.top_k, compute_ablations=req.compute_ablations)
        log_memory_usage("at end of /api/analyze request")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/ablate-head", response_model=AblateHeadResp)
def ablate_head(req: AblateHeadReq):
    log_memory_usage("at start of /api/ablate-head request")
    try:
        result = ablate_head_analysis(
            text=req.text,
            position=req.position,
            model_name=req.model_name,
            layer=req.layer,
            head=req.head,
            top_k=req.top_k,
        )
        log_memory_usage("at end of /api/ablate-head request")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class CompositionScoresResp(BaseModel):
    q_composition: List[List[float]]
    k_composition: List[List[float]]
    v_composition: List[List[float]]


@app.get("/api/composition-scores")
def get_composition_scores(model_name: str = "t2"):
    """
    Get Q-, K-, and V-composition scores for a model.
    Only works for multi-layer models (t2).
    Returns matrices of shape [layer1_head, layer0_head].
    """
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    model = MODELS[model_name]

    if model.cfg.n_layers < 2:
        raise HTTPException(status_code=400, detail=f"Model {model_name} has only {model.cfg.n_layers} layer(s). Need at least 2 for composition analysis.")

    scores = calculate_composition_scores(model)
    return CompositionScoresResp(**scores)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
