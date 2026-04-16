"""
Training script for Mini Transformer LM.

Usage (from project root):
    python -m src.train
    python -m src.train --config train_config.yaml
"""

import argparse
import json
import math
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

# Allow running as both `python src/train.py` and `python -m src.train`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import prepare_data
from src.model import MiniTransformer, ModelConfig


def get_lr(step: int, config: dict) -> float:
    """Cosine decay with linear warmup."""
    warmup = config["training"]["warmup_iters"]
    max_lr = config["training"]["learning_rate"]
    min_lr = config["training"]["min_lr"]
    decay_iters = config["training"]["lr_decay_iters"]

    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step > decay_iters:
        return min_lr
    progress = (step - warmup) / (decay_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, val_loader, eval_iters: int, device) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def train(config_path: str = "train_config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    ckpt_dir = config["checkpoint"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    tokenizer, train_ds, val_ds = prepare_data(config)
    bs = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=True)

    # Model
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config["model"]["block_size"],
        n_embd=config["model"]["n_embd"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
    )
    model = MiniTransformer(cfg).to(device)
    print(f"Parameters: {model.num_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Save tokenizer alongside checkpoints
    tokenizer.save(os.path.join(ckpt_dir, "tokenizer.json"))

    training_log: list[dict] = []
    best_val_loss = float("inf")
    max_iters = config["training"]["max_iters"]
    eval_interval = config["training"]["eval_interval"]
    eval_iters = config["training"]["eval_iters"]
    save_interval = config["checkpoint"]["save_interval"]
    log_file = config["logging"]["log_file"]

    train_iter = iter(train_loader)
    val_loss = float("inf")

    for step in range(max_iters):
        # Fetch batch (cycle through loader)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Update learning rate
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
        optimizer.step()

        # Evaluation
        if step % eval_interval == 0 or step == max_iters - 1:
            val_loss = estimate_loss(model, val_loader, eval_iters, device)
            ppl = math.exp(min(val_loss, 20))  # cap to avoid overflow
            entry = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "perplexity": round(ppl, 2),
                "lr": round(lr, 8),
            }
            training_log.append(entry)
            with open(log_file, "w") as f:
                json.dump(training_log, f, indent=2)

            print(
                f"step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f} "
                f"| ppl {ppl:.1f} | lr {lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": cfg,
                        "val_loss": val_loss,
                    },
                    os.path.join(ckpt_dir, "best.pt"),
                )
                print(f"  -> saved best checkpoint (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if step > 0 and step % save_interval == 0:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "val_loss": val_loss,
                },
                os.path.join(ckpt_dir, f"checkpoint_{step}.pt"),
            )

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.yaml")
    args = parser.parse_args()
    train(args.config)
