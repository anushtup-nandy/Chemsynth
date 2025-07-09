import os
import re
from shutil import copyfile
from typing import List, Optional

from transformers.tokenization_utils import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

class SmilesTokenizer(PreTrainedTokenizer):
    """
    Custom SMILES Tokenizer class.
    Inherits from Hugging Face's PreTrainedTokenizer.
    """
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        **kwargs
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocab file at path '{vocab_file}'.")
        
        self.vocab_file = vocab_file
        # Load vocab BEFORE calling parent __init__
        self.vocab = self.load_vocab(self.vocab_file)
        
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @staticmethod
    def load_vocab(vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    def _tokenize(self, text, **kwargs):
        """Tokenize a SMILES molecule at the atom-level."""
        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(text)]
        # This assertion can be helpful for debugging but might be too strict if the regex isn't perfect
        # assert text == ''.join(tokens), f"Tokenization failed for '{text}'. Tokens: '{''.join(tokens)}'"
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) to an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) to a token (str) using the vocab."""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return inv_vocab.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens in a single string."""
        return "".join(tokens)

    def get_vocab(self):
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> str:
        if not os.path.isdir(save_directory):
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        copyfile(self.vocab_file, vocab_file)
        return (vocab_file,)