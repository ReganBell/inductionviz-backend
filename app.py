from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional
import os
from pathlib import Path
from collections import defaultdict
import psutil
import logging
import threading

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import numpy as np

from bigram_model import load_bigram_model, SparseBigramModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log_memory_usage(context: str = ""):
    """Log current memory usage in MB"""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    logger.info(f"Memory usage {context}: {mem_mb:.2f} MB")
    return mem_mb


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

# Locks to ensure thread-safe model access
MODEL_LOCKS: Dict[str, threading.Lock] = {
    "t1": threading.Lock(),
    "t2": threading.Lock(),
}

log_memory_usage("after loading both models")

# Use the model's tokenizer (NeoX, not GPT-2)
enc = MODELS["t1"].tokenizer
VOCAB = MODELS["t1"].cfg.d_vocab


def encode(text: str) -> List[int]:
    return enc.encode(text)


def decode(ids: List[int]) -> str:
    return enc.decode(ids)

# Load bigram model
BIGRAM_MODEL_PATH = Path(__file__).parent / "openwebtext_bigram_counts_neox.pkl"

# Download from R2 if not present (for Railway deployment)
BIGRAM_MODEL_URL = os.getenv("BIGRAM_MODEL_URL")
if BIGRAM_MODEL_URL and not BIGRAM_MODEL_PATH.exists():
    logger.info(f"Bigram model not found locally, downloading from R2: {BIGRAM_MODEL_URL}")
    try:
        import urllib.request
        import tempfile

        # Download to temp file first, then move (atomic operation)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            logger.info("Downloading bigram model (291MB, this may take a minute)...")
            urllib.request.urlretrieve(BIGRAM_MODEL_URL, tmp_file.name)

            # Move to final location
            import shutil
            shutil.move(tmp_file.name, BIGRAM_MODEL_PATH)
            logger.info(f"Bigram model downloaded successfully to {BIGRAM_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to download bigram model: {e}")
        logger.warning("Bigram endpoints will return 503 errors")

BIGRAM_MODEL: Optional[SparseBigramModel] = load_bigram_model(BIGRAM_MODEL_PATH)

if BIGRAM_MODEL:
    logger.info("Loaded OpenWebText bigram model")
    log_memory_usage("after loading bigram model")
else:
    logger.warning(f"Bigram model not found at {BIGRAM_MODEL_PATH}")


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


def _get_bigram_predictions(token_id: int, k: int = 10) -> Optional[List[Dict[str, object]]]:
    """
    Get top-k bigram predictions for a given token ID using the SparseBigramModel.

    Returns None if bigram model is not loaded or token not in model.
    Returns list of prediction dicts with token, id, prob, logit fields.
    """
    if not BIGRAM_MODEL:
        return None

    # Get top-k predictions from the bigram model
    top_k = BIGRAM_MODEL.get_top_k_next(token_id, k=k)

    if len(top_k) == 0:
        return None

    # Format predictions
    predictions = []
    for next_token_id, prob in top_k:
        token_str = decode([next_token_id])
        logit = float(np.log(prob)) if prob > 0 else float('-inf')
        predictions.append({
            "token": token_str,
            "id": int(next_token_id),
            "prob": float(prob),
            "logit": logit,
        })

    return predictions


def analyze_text(text: str, *, top_k: int = 10, compute_ablations: bool = False, models: Optional[Dict[str, HookedTransformer]] = None):
    if models is None:
        models = MODELS
    # load text from hp.txt
    # with open('hp.txt', 'r') as f:
    #     text = f.read()
    ids = encode(text)

    # Prepend BOS token (consistent with attention-patterns endpoint)
    bos_token_id = models["t1"].tokenizer.bos_token_id
    if bos_token_id is not None and (len(ids) == 0 or ids[0] != bos_token_id):
        ids = [bos_token_id] + ids

    if len(ids) < 2:
        raise ValueError("Need at least two tokens")
    toks = torch.tensor(ids, device=DEVICE, dtype=torch.long).unsqueeze(0)
    T = toks.size(1)

    logits1_full, cache1 = _run_with_cache(models["t1"], toks)
    logits2_full, cache2 = _run_with_cache(models["t2"], toks)

    skip_positions = set(_skip_trigram_positions(ids))

    tokens_info = [{"id": int(i), "text": decode([int(i)])} for i in ids]
    positions = []

    for t in range(1, T):
        prev_id = ids[t - 1]
        next_id = ids[t]

        # Get bigram predictions using the shared function
        bigram_predictions = _get_bigram_predictions(prev_id, k=top_k)
        bigram_available = bigram_predictions is not None

        l1 = logits1_full[t - 1]
        l2 = logits2_full[t - 1]

        attn1 = [[[] for _ in range(models["t1"].cfg.n_heads)] for _ in range(models["t1"].cfg.n_layers)]
        attn2 = [[[] for _ in range(models["t2"].cfg.n_heads)] for _ in range(models["t2"].cfg.n_layers)]

        for layer in range(models["t1"].cfg.n_layers):
            for head in range(models["t1"].cfg.n_heads):
                attn1[layer][head] = cache1[f"blocks.{layer}.attn.hook_pattern"][0, head, t, :t].detach().cpu().tolist()
        for layer in range(models["t2"].cfg.n_layers):
            for head in range(models["t2"].cfg.n_heads):
                attn2[layer][head] = cache2[f"blocks.{layer}.attn.hook_pattern"][0, head, t, :t].detach().cpu().tolist()


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
        if bigram_available and bigram_predictions:
            # Find the actual next token in the predictions to get its probability
            next_token_prob = None
            for pred in bigram_predictions:
                if pred["id"] == next_id:
                    next_token_prob = pred["prob"]
                    break
            # If token not in top-k, get it directly
            if next_token_prob is None:
                full_preds = _get_bigram_predictions(prev_id, k=VOCAB)
                if full_preds:
                    for pred in full_preds:
                        if pred["id"] == next_id:
                            next_token_prob = pred["prob"]
                            break
            loss_bigram = float(-np.log(next_token_prob)) if next_token_prob and next_token_prob > 0 else None
        else:
            loss_bigram = None

        loss_t1 = float(-F.log_softmax(l1, dim=-1)[next_id].item())
        loss_t2 = float(-F.log_softmax(l2, dim=-1)[next_id].item())

        # Prepare topk data - handle missing bigram data
        topk_data = {
            "t1": _topk_pack(l1, top_k),
            "t2": _topk_pack(l2, top_k),
            "bigram": bigram_predictions if bigram_available else None,
        }

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



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Induction Heads API"}



@torch.no_grad()
def get_attention_patterns(
    text: str,
    model_name: str = "t2",
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    compute_ov: bool = True,
    compute_full: bool = True,
    top_k: int = 10,
    normalize_ov: bool = True,
):
    """
    Get attention patterns for a given text.

    Returns attention weights showing which tokens each position attends to.
    Optionally computes OV circuit predictions for each token.
    Optionally computes full model predictions (with attention) for each token.

    Returns:
        tokens: List of {id, text} dicts
        attention: [position][layer][head][src_position] - attention weights
        ov_predictions: [token][layer][head] -> list of top-k {token, logit} predictions
        full_predictions: [token] -> list of top-k {token, id, logit} predictions (full model with attention)
        model_name: Which model was used
        n_layers: Number of layers returned
        n_heads: Number of heads returned
    """
    logger.info(f"get_attention_patterns called with text='{text}', model_name={model_name}, compute_full={compute_full}, top_k={top_k}")

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model = MODELS[model_name]
    model_lock = MODEL_LOCKS[model_name]

    ids = encode(text)
    logger.info(f"Encoded to {len(ids)} tokens: {ids}")

    # Prepend BOS token
    bos_token_id = model.tokenizer.bos_token_id
    if bos_token_id is not None and (len(ids) == 0 or ids[0] != bos_token_id):
        ids = [bos_token_id] + ids

    if len(ids) < 2:
        raise ValueError("Need at least two tokens to compute attention")

    toks = torch.tensor(ids, device=DEVICE, dtype=torch.long).unsqueeze(0)
    T = toks.size(1)

    # Use lock to ensure thread-safe model access
    with model_lock:
        # Reset any existing hooks to ensure clean state
        model.reset_hooks()

        # Run model with cache to get attention patterns and logits
        # For OV circuit, we only need attention patterns - we'll use W_E directly
        hook_names = [f"blocks.{layer}.attn.hook_pattern" for layer in range(model.cfg.n_layers)]

        logits, cache = model.run_with_cache(
            toks,
            names_filter=lambda n: n in hook_names,
            return_type="logits",
        )
        logger.info(f"Got logits with shape: {logits.shape}, toks shape: {toks.shape}")

        # Log cache contents
        for key in cache.keys():
            logger.info(f"Cache key: {key}, shape: {cache[key].shape}")

        # Filter layers and heads if specified
        layer_indices = layers if layers is not None else list(range(model.cfg.n_layers))
        head_indices = heads if heads is not None else list(range(model.cfg.n_heads))

        # Extract tokens
        tokens_info = [{"id": int(i), "text": decode([int(i)])} for i in ids]

        # Extract attention patterns for each position (must be done inside lock while cache is valid)
        attention_by_position = []
        for pos in range(1, T):  # Start from 1 since position 0 has no previous tokens
            position_attn = []
            for layer in layer_indices:
                layer_attn = []
                for head in head_indices:
                    # Get attention pattern: shape [pos+1] (attention to positions 0..pos)
                    cache_key = f"blocks.{layer}.attn.hook_pattern"
                    if cache_key not in cache:
                        logger.error(f"Cache key {cache_key} not found! Available keys: {list(cache.keys())}")
                        raise ValueError(f"Cache missing expected key: {cache_key}")

                    attn_tensor = cache[cache_key]
                    logger.info(f"Accessing {cache_key} with shape {attn_tensor.shape}, pos={pos}, T={T}")

                    if pos >= attn_tensor.shape[2]:
                        logger.error(f"Position {pos} out of bounds for tensor shape {attn_tensor.shape}")
                        raise ValueError(f"Position {pos} >= cache dimension {attn_tensor.shape[2]}")

                    attn_pattern = attn_tensor[0, head, pos, :pos+1]
                    layer_attn.append(attn_pattern.detach().cpu().tolist())
                position_attn.append(layer_attn)
            attention_by_position.append(position_attn)

    # Compute OV circuit predictions if requested
    ov_predictions = None
    if compute_ov:
        # Get residual stream after embedding
        # We'll use hook_embed for position 0, and hook_resid_pre for other positions
        ov_predictions = []

        for t in range(T):
            token_ov = []

            # Get token embedding directly (matching attention_head_analysis.py methodology)
            # This computes OV circuit through the embedding matrix, not the residual stream
            token_id = ids[t]
            token_embedding = model.W_E[token_id]  # [d_model]

            for layer in layer_indices:
                layer_ov = []
                for head in head_indices:
                    # Apply W_V to get value vector
                    W_V = model.W_V[layer, head]  # [d_model, d_head]
                    value = token_embedding @ W_V  # [d_head]

                    # Apply W_O to get output vector
                    W_O = model.W_O[layer, head]  # [d_head, d_model]
                    output = value @ W_O  # [d_model]

                    # Apply unembed to get logits
                    ov_logits = output @ model.W_U  # [vocab_size]

                    # Optionally normalize by subtracting mean
                    # Normalized: shows which tokens are boosted relative to average
                    # Non-normalized: shows raw logit boosts (useful for understanding absolute effects)
                    if normalize_ov:
                        ov_logits_for_topk = ov_logits - ov_logits.mean()
                    else:
                        ov_logits_for_topk = ov_logits

                    # Get top-k predictions (top 10)
                    top_k = 10
                    top_logits, top_indices = torch.topk(ov_logits_for_topk, top_k)

                    # Convert to list of {token, logit} dicts
                    predictions = [
                        {
                            "token": decode([int(idx)]),
                            "id": int(idx),
                            "logit": float(logit),
                        }
                        for idx, logit in zip(top_indices, top_logits)
                    ]
                    layer_ov.append(predictions)

                token_ov.append(layer_ov)

            ov_predictions.append(token_ov)

    # Compute full model predictions if requested
    full_predictions = None
    full_predictions_normalized = None
    if compute_full:
        logger.info(f"Computing full predictions with logits shape: {logits.shape}, T={T}")
        full_predictions = []
        full_predictions_normalized = []

        for t in range(T):
            position_logits = logits[0, t, :]  # [vocab_size]

            # Compute probabilities for actual model output
            probs = F.softmax(position_logits, dim=-1)

            # Normalize by subtracting mean for interpretable "attention boost"
            logits_normalized = position_logits - position_logits.mean()

            # Get top-k for both
            top_logits_norm, top_indices_norm = torch.topk(logits_normalized, top_k)
            top_probs, top_indices_prob = torch.topk(probs, top_k)

            # Normalized logits (attention boost)
            predictions_normalized = [
                {
                    "token": decode([int(idx)]),
                    "id": int(idx),
                    "logit": float(logit),
                }
                for idx, logit in zip(top_indices_norm, top_logits_norm)
            ]
            full_predictions_normalized.append(predictions_normalized)

            # Actual probabilities
            predictions_prob = [
                {
                    "token": decode([int(idx)]),
                    "id": int(idx),
                    "prob": float(prob),
                }
                for idx, prob in zip(top_indices_prob, top_probs)
            ]
            full_predictions.append(predictions_prob)

        logger.info(f"Computed {len(full_predictions)} full predictions")

    result = {
        "tokens": tokens_info,
        "attention": attention_by_position,
        "model_name": model_name,
        "n_layers": len(layer_indices),
        "n_heads": len(head_indices),
    }

    if ov_predictions is not None:
        result["ov_predictions"] = ov_predictions

    if full_predictions is not None:
        result["full_predictions"] = full_predictions
        result["full_predictions_normalized"] = full_predictions_normalized

    return result


@app.post("/api/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    log_memory_usage("at start of /api/analyze request")
    try:
        result = analyze_text(req.text, top_k=req.top_k, compute_ablations=req.compute_ablations)
        log_memory_usage("at end of /api/analyze request")
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


class AttentionPatternsReq(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    model_name: str = "t2"  # "t1" or "t2"
    layers: Optional[List[int]] = None  # e.g., [0, 1] or None for all
    heads: Optional[List[int]] = None   # e.g., [0, 2] or None for all
    normalize_ov: bool = True  # Whether to normalize OV logits by subtracting mean


class AttentionPatternsResp(BaseModel):
    tokens: List[Dict[str, object]]  # [{"id": int, "text": str}, ...]
    attention: List[List[List[List[float]]]]  # [position][layer][head][src_position]
    ov_predictions: Optional[List[List[List[List[Dict[str, object]]]]]] = None  # [token][layer][head][prediction]
    full_predictions: Optional[List[List[Dict[str, object]]]] = None  # [token][prediction] - full model predictions (probabilities)
    full_predictions_normalized: Optional[List[List[Dict[str, object]]]] = None  # [token][prediction] - normalized logits (attention boost)
    model_name: str
    n_layers: int
    n_heads: int


class BigramTopKReq(BaseModel):
    model_config = ConfigDict(extra="ignore")

    token: str  # Input token string (will be encoded to ID)
    k: int = 10  # Number of top predictions to return


class BigramTopKResp(BaseModel):
    predictions: List[Dict[str, object]]  # [{token, id, prob, logit}, ...]
    input_token: str
    input_token_id: int
    available: bool  # False if token not in model


class BigramBatchReq(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str  # Input text to tokenize and analyze
    k: int = 10  # Number of top predictions per token


class BigramBatchResp(BaseModel):
    tokens: List[Dict[str, object]]  # [{id, text}, ...] for all tokens
    predictions: List[List[Dict[str, object]]]  # [position][prediction] - predictions for each position


@app.post("/api/attention-patterns", response_model=AttentionPatternsResp)
def attention_patterns(req: AttentionPatternsReq):
    """
    Get attention patterns for a given text.

    Returns which tokens each position attends to.
    Much faster than full /api/analyze since it only computes attention.
    """
    try:
        result = get_attention_patterns(
            text=req.text,
            model_name=req.model_name,
            layers=req.layers,
            heads=req.heads,
            normalize_ov=req.normalize_ov,
        )
        return AttentionPatternsResp(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/bigram-topk", response_model=BigramTopKResp)
def bigram_topk(req: BigramTopKReq):
    """
    Get top-k bigram predictions for a given input token.

    Returns the most likely next tokens according to the OpenWebText
    bigram model trained on 100k documents.
    """
    if not BIGRAM_MODEL:
        raise HTTPException(
            status_code=503,
            detail="Bigram model not loaded. Check server logs."
        )

    # Encode the input token
    token_ids = encode(req.token)

    if len(token_ids) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty token string"
        )

    # Use the first token ID if multiple tokens
    token_id = token_ids[0]

    if len(token_ids) > 1:
        logger.warning(
            f"Input token '{req.token}' encoded to {len(token_ids)} tokens. "
            f"Using first token ID: {token_id}"
        )

    # Get top-k predictions using the shared function
    predictions = _get_bigram_predictions(token_id, k=req.k)

    return BigramTopKResp(
        predictions=predictions if predictions else [],
        input_token=req.token,
        input_token_id=token_id,
        available=predictions is not None and len(predictions) > 0
    )


@app.post("/api/bigram-batch", response_model=BigramBatchResp)
def bigram_batch(req: BigramBatchReq):
    """
    Get top-k bigram predictions for all tokens in the input text.

    Tokenizes the input text and returns predictions for each token position.
    For position i, returns the top-k most likely tokens to follow token[i].
    """
    if not BIGRAM_MODEL:
        raise HTTPException(
            status_code=503,
            detail="Bigram model not loaded. Check server logs."
        )

    # Encode the input text
    token_ids = encode(req.text)

    # Prepend BOS token (consistent with attention-patterns endpoint)
    bos_token_id = MODELS["t1"].tokenizer.bos_token_id
    if bos_token_id is not None and (len(token_ids) == 0 or token_ids[0] != bos_token_id):
        token_ids = [bos_token_id] + token_ids

    if len(token_ids) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty text string"
        )

    # Get token information
    tokens = [{"id": int(tid), "text": decode([tid])} for tid in token_ids]

    # Get predictions for each token using the shared function
    all_predictions = []
    for token_id in token_ids:
        predictions = _get_bigram_predictions(token_id, k=req.k)
        # If no predictions available, append empty list
        all_predictions.append(predictions if predictions else [])

    return BigramBatchResp(
        tokens=tokens,
        predictions=all_predictions
    )


class AttentionTopKReq(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    model_name: str = "t1"  # "t1" or "t2"
    position: int  # Which token position to get predictions for
    k: int = 10  # Number of top predictions to return


class AttentionTopKResp(BaseModel):
    predictions: List[Dict[str, object]]  # [{token, id, logit}, ...]
    position: int
    token: str


@app.post("/api/attention-topk", response_model=AttentionTopKResp)
def attention_topk(req: AttentionTopKReq):
    """
    Get top-k predictions from full model (with attention) for a specific position.

    Returns the most likely next tokens after running the full forward pass
    with attention, normalized by subtracting mean for interpretability.
    """
    if req.model_name not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_name}")

    model = MODELS[req.model_name]
    ids = encode(req.text)

    # Prepend BOS token
    bos_token_id = model.tokenizer.bos_token_id
    if bos_token_id is not None and (len(ids) == 0 or ids[0] != bos_token_id):
        ids = [bos_token_id] + ids

    if req.position < 0 or req.position >= len(ids):
        raise HTTPException(status_code=400, detail=f"Invalid position {req.position}")

    toks = torch.tensor(ids, device=DEVICE, dtype=torch.long).unsqueeze(0)

    # Run model to get logits
    logits = model(toks)  # [1, seq_len, vocab_size]
    position_logits = logits[0, req.position, :]  # [vocab_size]

    # Normalize by subtracting mean (same as OV circuit normalization)
    logits_normalized = position_logits - position_logits.mean()

    # Get top-k
    top_logits, top_indices = torch.topk(logits_normalized, req.k)

    predictions = [
        {
            "token": decode([int(idx)]),
            "id": int(idx),
            "logit": float(logit),
        }
        for idx, logit in zip(top_indices, top_logits)
    ]

    return AttentionTopKResp(
        predictions=predictions,
        position=req.position,
        token=decode([ids[req.position]]) if req.position < len(ids) else ""
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
