import torch
from tqdm import tqdm
import numpy as np

class Dataset:
    def __init__(self, bos_char='S', eos_char='E', mask_char='M') -> None:
        self.bos_char = bos_char
        self.eos_char = eos_char
        self.mask_char = mask_char        
        self.char2num = None
        self.num2char = None
        self.x = None
        self.z = None

    def load_translation_dataset(self, file_path='spa-eng/spa.txt', max_len = 20):
        """
        Loads dataset with language2language sentences 
        returns:
            (z, x): list of tensors with source and target tensor sentences, respectively
            encoder: maps string/list of characters to list of tokens
            decoder: maps list of tokens to list of characters 
        """
        vocab_z, vocab_x = set(), set()
        z, x = [], []
        with open(file_path) as f:
            for line in tqdm(f):
                parts = line.strip().lower().split('\t')
                # (max_len - 2) to account for begin and end of sequence tokens
                if len(parts[0]) > (max_len - 2) or len(parts[1]) > (max_len - 2):
                    continue
                z.append(parts[0])
                vocab_z.update(parts[0])
                x.append(parts[1])
                vocab_x.update(parts[1])

        all_chars = vocab_x.union(vocab_z)
        self.char2num = {char:num for num, char in enumerate(sorted(all_chars))}
        start_token = len(all_chars)
        end_token = len(all_chars) + 1
        self.char2num[self.bos_char] = start_token
        self.char2num[self.eos_char] = end_token
        self.num2char = {n:c for c, n in self.char2num.items()}
        self.z = [torch.tensor([start_token] + self.tokenize(line) + [end_token]) for line in z]
        self.x = [torch.tensor([start_token] + self.tokenize(line) + [end_token]) for line in x]
        print(f"*** Dataset size: {len(x)}. Vocabulary size: {len(self.char2num)}")

    def load_book_dataset(self, file_path='crime_and_punishment.txt', max_len=20):
        """
        Loads book dataset.
        """    
        x = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().lower()
        all_chars = set(text)
        self.char2num = {char:num for num, char in enumerate(sorted(all_chars))}
        dataset_size = 10000
        start_ix = torch.randint(len(text) - max_len, (dataset_size,))
        x = [text[ix:ix+max_len] for ix in start_ix]
        mask_token = len(all_chars)
        self.char2num[self.mask_char] = mask_token
        self.num2char = {n:c for c, n in self.char2num.items()}
        self.x = torch.stack([torch.tensor(self.tokenize(sample)) for sample in x])
        print(f"*** Dataset size: {self.x.shape[0]}. Vocabulary size: {len(self.char2num)}")
        

    def tokenize(self, chars):
        return [self.char2num[c] for c in chars]
    
    def detokenize(self, tokens):
        return [self.num2char[n] for n in tokens]
    
    def vocabulary_size(self):
        return len(self.char2num)