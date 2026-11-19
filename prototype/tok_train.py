import argparse
from pathlib import Path

from data import download_shakespeare
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--data", type=str, default=str(ROOT / "data/tinyshakespeare/input.txt"))
    parser.add_argument("--out", type=str, default=str(ROOT / "data/tinyshakespeare/tokenizer.json"))
    args = parser.parse_args()

    text = download_shakespeare(Path(args.data)).read_text(encoding="utf-8")
    print(f"training BPE on {len(text):,} chars, vocab_size={args.vocab_size:,}")
    tokenizer = BPETokenizer()
    tokenizer.train(text, args.vocab_size)
    tokenizer.save(args.out)
    print(f"saved {args.out}, vocab={tokenizer.vocab_size:,}, bytes/token={tokenizer.bytes_per_token(text[:20000]):.3f}")


if __name__ == "__main__":
    main()