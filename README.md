# mini-transformer-lm

A decoder-only transformer language model trained on Shakespeare, with character-level and BPE tokenization comparison.

## Setup

```bash
pip install -r requirements.txt
```

```bash
curl -o data/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Train

```bash
# character-level (default)
python -m src.train

# BPE
python -m src.train --config train_config_bpe.yaml
```

## Run the app

```bash
uvicorn api.main:app --reload --port 8000
```

Open [http://localhost:8000/app](http://localhost:8000/app)

- **Generate** — prompt the model with temperature and top-k controls
- **Attention** — per-layer, per-head attention heatmaps
- **Training** — loss and perplexity curves
- **Model Info** — architecture details
