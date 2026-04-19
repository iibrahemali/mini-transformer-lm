"""
FastAPI backend for Mini Transformer LM.

Run from project root:
    uvicorn api.main:app --reload --port 8000

Then open: http://localhost:8000/app
"""

import json
import os
import sys

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_tokenizer
from src.generate import generate_text, generate_with_confidence
from src.model import MiniTransformer

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Mini Transformer LM", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model: MiniTransformer | None = None
_tokenizer = None
_active_model_key: str = "char"

MODELS = {
    "char": {
        "checkpoint": "checkpoints/char/best.pt",
        "tokenizer":  "checkpoints/char/tokenizer.json",
        "log":        "checkpoints/char/training_log.json",
    },
    "bpe": {
        "checkpoint": "checkpoints/bpe/best.pt",
        "tokenizer":  "checkpoints/bpe/tokenizer.json",
        "log":        "checkpoints/bpe/training_log.json",
    },
}


def _load_model(key: str = None) -> bool:
    global _model, _tokenizer, _active_model_key
    if key:
        _active_model_key = key
    paths = MODELS[_active_model_key]
    if not os.path.exists(paths["checkpoint"]) or not os.path.exists(paths["tokenizer"]):
        return False
    try:
        _tokenizer = load_tokenizer(paths["tokenizer"])
        ckpt = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        _model = MiniTransformer(ckpt["config"])
        _model.load_state_dict(ckpt["model_state_dict"])
        _model.eval()
        print(f"Loaded [{_active_model_key}] — {_model.num_params / 1e6:.2f}M params")
        return True
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        return False


@app.on_event("startup")
async def startup():
    # Try char first, fall back to bpe
    if not _load_model("char"):
        _load_model("bpe")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1024)
    max_tokens: int = Field(default=200, ge=1, le=1000)
    temperature: float = Field(default=0.8, ge=0.01, le=2.0)
    top_k: int = Field(default=40, ge=1, le=200)


class AttentionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)


class SwitchModelRequest(BaseModel):
    model: str  # "char" or "bpe"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/switch-model")
async def switch_model(req: SwitchModelRequest):
    if req.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'. Choose 'char' or 'bpe'.")
    ok = _load_model(req.model)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No checkpoint found for '{req.model}'. Train it first.")
    return {"status": "switched", "active": _active_model_key, "num_params": _model.num_params}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    result = generate_text(_model, _tokenizer, req.prompt, req.max_tokens, req.temperature, req.top_k)
    return {"generated": result}


@app.post("/generate-with-confidence")
async def generate_confidence(req: GenerateRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    tokens = generate_with_confidence(
        _model, _tokenizer, req.prompt, req.max_tokens, req.temperature, req.top_k
    )
    max_entropy = max((t["entropy"] for t in tokens), default=1.0) or 1.0
    return {"prompt": req.prompt, "tokens": tokens, "max_entropy": round(max_entropy, 4)}


@app.post("/attention")
async def get_attention(req: AttentionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    block_size = _model.config.block_size
    text = req.text[:block_size]
    tokens = _tokenizer.encode(text)

    if not tokens:
        raise HTTPException(status_code=400, detail="Text contains no known characters.")

    idx = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        _model(idx)

    weights = _model.get_attention_weights()
    token_labels = [_tokenizer.itos.get(t, "?") for t in tokens]

    return {
        "tokens": token_labels,
        "n_layers": len(weights),
        "n_heads": _model.config.n_heads,
        "attention_weights": weights,
    }


@app.get("/training-logs")
async def get_training_logs():
    log_path = MODELS[_active_model_key]["log"]
    if not os.path.exists(log_path):
        return {"logs": [], "message": "No training log found. Run training first."}
    with open(log_path) as f:
        logs = json.load(f)
    return {"logs": logs}


@app.get("/model-info")
async def get_model_info():
    if _model is None:
        return {"loaded": False, "message": "No checkpoint found. Train the model first."}

    cfg = _model.config
    return {
        "loaded": True,
        "active_tokenizer": _active_model_key,
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size,
        "n_embd": cfg.n_embd,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "dropout": cfg.dropout,
        "num_params": _model.num_params,
        "num_params_fmt": f"{_model.num_params / 1e6:.2f}M",
    }


@app.post("/reload")
async def reload_model():
    ok = _load_model()
    if not ok:
        raise HTTPException(status_code=404, detail="No checkpoint found.")
    return {"status": "reloaded", "active": _active_model_key}


# ---------------------------------------------------------------------------
# Serve frontend (must be registered after all API routes)
# ---------------------------------------------------------------------------

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")
