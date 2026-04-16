"""
Text generation utilities.

Usage (from project root):
    python -m src.generate --prompt "To be or not" --max_tokens 300
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    device = next(model.parameters()).device
    block_size = model.config.block_size

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]

    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

    return tokenizer.decode(idx[0].tolist())


def load_model_from_checkpoint(ckpt_path: str, tokenizer_path: str):
    from src.model import MiniTransformer
    from src.data import CharTokenizer

    tokenizer = CharTokenizer.load(tokenizer_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MiniTransformer(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    args = parser.parse_args()

    model, tokenizer = load_model_from_checkpoint(args.checkpoint, args.tokenizer)
    output = generate_text(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k)
    print(output)
