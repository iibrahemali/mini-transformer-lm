"""
Tokenizer comparison: CharTokenizer vs BPETokenizer.

Usage (from project root):
    python compare_tokenizers.py
    python compare_tokenizers.py --data data/shakespeare.txt --bpe_sizes 500 1000 2000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import CharTokenizer, BPETokenizer

SAMPLE_TEXT = (
    "ROMEO:\nBut, soft! what light through yonder window breaks?\n"
    "It is the east, and Juliet is the sun.\n"
    "Arise, fair sun, and kill the envious moon,\n"
    "Who is already sick and pale with grief.\n"
)


def analyse(name: str, tokenizer, text: str, sample: str) -> dict:
    t0 = time.perf_counter()
    tokens = tokenizer.encode(text)
    encode_sec = time.perf_counter() - t0

    sample_tokens = tokenizer.encode(sample)
    sample_decoded = tokenizer.decode(sample_tokens)

    return {
        "name": name,
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": len(tokens),
        "compression": len(text) / len(tokens),   # chars per token
        "encode_sec": encode_sec,
        "sample_tokens": sample_tokens,
        "sample_decoded": sample_decoded,
    }


def print_table(results: list[dict], text_len: int):
    header = f"{'Tokenizer':<22} {'Vocab':>6} {'Tokens':>10} {'Chars/tok':>10} {'Encode(s)':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<22} {r['vocab_size']:>6,} {r['total_tokens']:>10,} "
            f"{r['compression']:>10.3f} {r['encode_sec']:>10.3f}"
        )
    print("=" * len(header))
    print(f"\nSource text: {text_len:,} characters\n")


def print_sample_encodings(results: list[dict], sample: str):
    print(f"Sample text ({len(sample)} chars):")
    print(f"  {repr(sample[:80])}{'...' if len(sample) > 80 else ''}\n")
    for r in results:
        toks = r["sample_tokens"]
        decoded = r["sample_decoded"]
        print(f"  [{r['name']}]  {len(toks)} tokens")
        tok_strings = []
        # Show first 12 tokens as readable subwords
        for t in toks[:12]:
            tok_strings.append(repr(r.get("_tokenizer").itos.get(t, "?")))
        print(f"    tokens: [{', '.join(tok_strings)}{'...' if len(toks) > 12 else ''}]")
        match = "✓" if decoded == sample else "✗"
        print(f"    roundtrip: {match}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/shakespeare.txt")
    parser.add_argument(
        "--bpe_sizes", nargs="+", type=int, default=[500, 1000, 2000],
        help="BPE vocabulary sizes to compare",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Dataset not found at '{args.data}'.")
        print("Download it with:")
        print("  curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        sys.exit(1)

    with open(args.data, encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded '{args.data}': {len(text):,} characters\n")

    results = []

    # --- CharTokenizer ---
    print("Building CharTokenizer...", end=" ", flush=True)
    char_tok = CharTokenizer(text)
    print(f"done ({char_tok.vocab_size} chars)")
    r = analyse("char", char_tok, text, SAMPLE_TEXT)
    r["_tokenizer"] = char_tok
    results.append(r)

    # --- BPETokenizer at each requested vocab size ---
    for vs in args.bpe_sizes:
        print(f"Training BPETokenizer (vocab={vs})...")
        bpe_tok = BPETokenizer(text, vocab_size=vs)
        r = analyse(f"bpe-{vs}", bpe_tok, text, SAMPLE_TEXT)
        r["_tokenizer"] = bpe_tok
        results.append(r)

    print_table(results, len(text))

    print_sample_encodings(results, SAMPLE_TEXT)

    # Sequence-length impact for a fixed block_size
    block_size = 256
    print(f"Effective context (block_size={block_size}):")
    for r in results:
        chars_per_tok = r["compression"]
        effective_chars = block_size * chars_per_tok
        effective_words = effective_chars / 5  # rough avg word length
        print(f"  {r['name']:<22} ~{effective_chars:,.0f} chars  (~{effective_words:,.0f} words) per context window")
    print()


if __name__ == "__main__":
    main()
