import json
import re
from pathlib import Path
from collections import Counter

# GPT-2 pretokenizer
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


def _pair_counts(words):
    counts = Counter()
    for word, freq in words.items():
        for pos in range(len(word) - 1):
            counts[(word[pos], word[pos + 1])] += freq
    return counts


def _merge(words, pair, new_id):
    merged = {}
    first, second = pair
    for word, freq in words.items():
        merged_word = []
        pos = 0
        while pos < len(word):
            if pos < len(word) - 1 and word[pos] == first and word[pos + 1] == second:
                merged_word.append(new_id)
                pos += 2
            else:
                merged_word.append(word[pos])
                pos += 1
        key = tuple(merged_word)
        merged[key] = merged.get(key, 0) + freq
    return merged


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {byte: bytes([byte]) for byte in range(256)}
        self.special_to_id = {}
        self.id_to_special = {}

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_to_id)

    def bos_id(self):
        return self.special_to_id["<|bos|>"]

    def train(self, text, vocab_size, verbose=True):
        n_special = len(SPECIAL_TOKENS)
        assert vocab_size >= 256 + n_special, f"vocab_size must be >= {256 + n_special}"
        n_merges = vocab_size - 256 - n_special

        words = Counter()
        for chunk in GPT2_PAT.findall(text):
            words[tuple(chunk.encode("utf-8"))] += 1

        merges = {}
        vocab = {byte: bytes([byte]) for byte in range(256)}
        for merge_index in range(n_merges):
            counts = _pair_counts(words)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + merge_index
            words = _merge(words, pair, new_id)
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (merge_index < 5 or (merge_index + 1) % 100 == 0 or merge_index + 1 == n_merges):
                print(f"merge {merge_index + 1:4d}/{n_merges} | {str(pair):<10} -> {new_id:4d} | freq: {counts[pair]:,}")

        self.merges = merges
        self.vocab = vocab
        self.special_to_id = {tok: 256 + n_merges + offset for offset, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_special = {token_id: tok for tok, token_id in self.special_to_id.items()}

    def _encode_word(self, word):
        ids = list(word)
        while len(ids) >= 2:
            pairs = [(self.merges.get((ids[pos], ids[pos + 1]), 10**18), pos) for pos in range(len(ids) - 1)]
            rank, pos = min(pairs)
            if rank == 10**18:
                break
            ids = ids[:pos] + [rank] + ids[pos + 2:]
        return ids

    def encode(self, text, add_bos=False):
        special_pat = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")") # keep atomic
        ids = []
        for part in special_pat.split(text):
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
                continue
            for chunk in GPT2_PAT.findall(part):
                ids.extend(self._encode_word(tuple(chunk.encode("utf-8"))))
        if add_bos:
            ids = [self.bos_id()] + ids
        return ids

    def decode(self, ids, skip_special=False):
        raw = bytearray()
        for token_id in ids:
            if token_id in self.id_to_special:
                if not skip_special:
                    raw.extend(self.id_to_special[token_id].encode("utf-8"))
                continue
            raw.extend(self.vocab[token_id])
        return raw.decode("utf-8", errors="replace")

    def bytes_per_token(self, text):
        ids = self.encode(text, add_bos=False)
        if not ids:
            return 0.0
        return len(text.encode("utf-8")) / len(ids)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "merges": [{"a": first, "b": second, "id": token_id} for (first, second), token_id in self.merges.items()],
            "special_tokens": self.special_to_id,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        merges = {}
        vocab = {byte: bytes([byte]) for byte in range(256)}
        for item in payload["merges"]:
            pair = (item["a"], item["b"])
            merges[pair] = item["id"]
            vocab[item["id"]] = vocab[item["a"]] + vocab[item["b"]]
        tokenizer.merges = merges
        tokenizer.vocab = vocab
        tokenizer.special_to_id = {name: int(token_id) for name, token_id in payload["special_tokens"].items()}
        tokenizer.id_to_special = {token_id: name for name, token_id in tokenizer.special_to_id.items()}
        return tokenizer