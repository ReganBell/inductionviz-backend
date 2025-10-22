import torch
from transformer_lens import HookedTransformer
import tiktoken

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

def encode(text: str):
    return enc.encode(text)

def decode(ids):
    return enc.decode(ids)

# Load model
print("Loading model...")
model = HookedTransformer.from_pretrained("attn-only-1l", device=DEVICE).to(DEVICE).eval()
print(f"Model has {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")

# Test text
text = "Mr and Mrs Dursley"
ids = encode(text)
print(f"\nText: {text}")
print(f"Tokens: {[decode([i]) for i in ids]}")
print(f"Token IDs: {ids}")

toks = torch.tensor(ids, device=DEVICE, dtype=torch.long).unsqueeze(0)
print(f"Tensor shape: {toks.shape}")

# Pick a position to analyze
position = 2  # Predicting token at position 3
next_token_id = ids[position + 1]
print(f"\n=== Analyzing position {position} ===")
print(f"Context tokens: {[decode([ids[i]]) for i in range(position + 1)]}")
print(f"Predicting token: '{decode([next_token_id])}' (id={next_token_id})")

# Get baseline logits
with torch.no_grad():
    logits_base = model(toks)[0, position]  # [vocab]
    print(f"\nBaseline logits shape: {logits_base.shape}")
    print(f"Baseline logit for correct token: {logits_base[next_token_id].item():.4f}")

    # Get top 5 predictions
    top5 = torch.topk(logits_base, 5)
    print("\nTop 5 baseline predictions:")
    for val, idx in zip(top5.values, top5.indices):
        print(f"  '{decode([int(idx)])}' (id={int(idx)}): {val.item():.4f}")

# Check available hook points
print("\n=== Available hook points ===")
for name, _ in model.named_modules():
    if 'attn' in name and 'hook' in name:
        print(f"  {name}")

# Test ablating head 0 in layer 0
layer = 0
head_to_ablate = 3
hook_name = f"blocks.{layer}.attn.hook_result"
print(f"\n=== Ablating L{layer}H{head_to_ablate} ===")
print(f"Using hook: {hook_name}")

def make_zero_head_hook(head_idx):
    def zero_head_hook(v, hook):
        print(f"Hook called! Hook name: {hook.name}, v.shape = {v.shape}")
        print(f"Zeroing head {head_idx}")
        v[:, :, head_idx, :] = 0.0
        return v
    return zero_head_hook

with torch.no_grad():
    output = model.run_with_hooks(
        toks,
        fwd_hooks=[(hook_name, make_zero_head_hook(head_to_ablate))]
    )
    print(f"Output type: {type(output)}")
    print(f"Output shape: {output.shape if hasattr(output, 'shape') else 'no shape'}")
    logits_ablated = output[0, position]

    print(f"Ablated logits shape: {logits_ablated.shape}")
    print(f"Ablated logit for correct token: {logits_ablated[next_token_id].item():.4f}")

    # Calculate delta
    delta_logits = logits_base - logits_ablated
    print(f"\nDelta logits shape: {delta_logits.shape}")
    print(f"Delta for correct token: {delta_logits[next_token_id].item():.4f}")

    # Top deltas
    top5_delta = torch.topk(delta_logits, 5)
    print("\nTop 5 tokens promoted by this head:")
    for val, idx in zip(top5_delta.values, top5_delta.indices):
        print(f"  '{decode([int(idx)])}' (id={int(idx)}): +{val.item():.4f}")

    top5_neg = torch.topk(-delta_logits, 5)
    print("\nTop 5 tokens suppressed by this head:")
    for val, idx in zip(top5_neg.values, top5_neg.indices):
        print(f"  '{decode([int(idx)])}' (id={int(idx)}): {-val.item():.4f}")

    # Check if hook was used
    print(f"\nMax absolute delta: {delta_logits.abs().max().item():.4f}")
    print(f"Sum of top 100 abs deltas: {torch.topk(delta_logits.abs(), min(100, delta_logits.size(0))).values.sum().item():.4f}")

print("\n=== Testing with hook_z instead ===")
with torch.no_grad():
    hook_name_z = f"blocks.{layer}.attn.hook_z"
    print(f"Using hook: {hook_name_z}")

    output_z = model.run_with_hooks(
        toks,
        fwd_hooks=[(hook_name_z, make_zero_head_hook(head_to_ablate))]
    )
    logits_ablated_z = output_z[0, position]
    delta_logits_z = logits_base - logits_ablated_z

    print(f"Delta for correct token (using hook_z): {delta_logits_z[next_token_id].item():.4f}")
    print(f"Max absolute delta (using hook_z): {delta_logits_z.abs().max().item():.4f}")
