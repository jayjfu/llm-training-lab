import tiktoken


class GPT2Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

    def bos_id(self):
        return self.enc.eot_token  # GPT-2 uses <|endoftext|> as the document separator

    def encode(self, text, add_bos=False):
        ids = self.enc.encode_ordinary(text)
        return [self.bos_id()] + ids if add_bos else ids

    def decode(self, ids, skip_special=False):
        return self.enc.decode([token_id for token_id in ids if not (skip_special and token_id == self.bos_id())])

    def token_bytes(self):
        lengths = [len(self.enc.decode_single_token_bytes(token_id)) for token_id in range(self.vocab_size)]
        lengths[self.bos_id()] = 0
        return lengths


TOKENIZERS = {"gpt2": GPT2Tokenizer}


def load_tokenizer(name):
    return TOKENIZERS[name]()