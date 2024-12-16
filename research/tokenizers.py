import torch
from typing import List, Tuple, Dict
import re
from collections import defaultdict, Counter

class ARNetTokenizer:
    def __init__(self, vocab_size: int = 32000, max_length: int = 512):
        """Initialize tokenizer with BPE-style subword tokenization"""
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.pad_token = 0
        self.unk_token = 1
        self.bos_token = 2
        self.eos_token = 3
        
        # Initialize vocabulary
        self.token_to_id: Dict[str, int] = {
            "<pad>": self.pad_token,
            "<unk>": self.unk_token,
            "<bos>": self.bos_token,
            "<eos>": self.eos_token,
        }
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        
        # Regex for basic tokenization
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")
        
    def train(self, texts: List[str], min_freq: int = 2):
        """Train tokenizer on texts using BPE-style subword tokenization"""
        # Get initial vocabulary from character-level tokens
        word_freqs = defaultdict(int)
        for text in texts:
            for match in self.pat.finditer(text):
                word = match.group()
                word_freqs[word] += 1
        
        # Filter by frequency and add to vocabulary
        for word, freq in word_freqs.items():
            if freq >= min_freq:
                for char in word:
                    if char not in self.token_to_id:
                        idx = len(self.token_to_id)
                        self.token_to_id[char] = idx
                        self.id_to_token[idx] = char
        
        # Add most frequent subwords until vocab_size is reached
        while len(self.token_to_id) < self.vocab_size:
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            self._merge_pair(*best_pair, word_freqs)
            
            # Add new merged pair to vocabulary
            if len(self.token_to_id) < self.vocab_size:
                new_token = ''.join(best_pair)
                if new_token not in self.token_to_id:
                    self.token_to_id[new_token] = len(self.token_to_id)
                    self.id_to_token[len(self.id_to_token)] = new_token
    
    def _get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs"""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            chars = list(word)
            for i in range(len(chars)-1):
                pairs[chars[i], chars[i+1]] += freq
        return pairs
    
    def _merge_pair(self, a: str, b: str, word_freqs: Dict[str, int]):
        """Merge pair of tokens in all words"""
        new_word_freqs = {}
        bigram = re.escape(a + b)
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in word_freqs:
            new_word = pattern.sub(a+b, word)
            new_word_freqs[new_word] = word_freqs[word]
        
        word_freqs.clear()
        word_freqs.update(new_word_freqs)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids"""
        tokens = []
        for match in self.pat.finditer(text):
            word = match.group()
            # Try to encode full word first
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                # Fallback to character-level encoding
                for char in word:
                    tokens.append(self.token_to_id.get(char, self.unk_token))
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Truncate if needed
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Pad if needed
        while len(tokens) < self.max_length:
            tokens.append(self.pad_token)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text"""
        tokens = []
        for idx in token_ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                if not skip_special_tokens or token not in ["<pad>", "<unk>", "<bos>", "<eos>"]:
                    tokens.append(token)
        return ''.join(tokens)
    
    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts"""
        # Encode all texts
        encoded_texts = [self.encode(text, add_special_tokens) for text in texts]
        
        # Convert to tensors
        padded = torch.tensor(encoded_texts, dtype=torch.long)
        lengths = torch.tensor([len(text) for text in texts])
        
        return padded, lengths
    
    def save(self, path: str):
        """Save tokenizer configuration"""
        config = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'special_tokens': {
                'pad': self.pad_token,
                'unk': self.unk_token,
                'bos': self.bos_token,
                'eos': self.eos_token
            }
        }
        torch.save(config, path)
    
    def load(self, path: str):
        """Load tokenizer configuration"""
        config = torch.load(path)
        self.vocab_size = config['vocab_size']
        self.max_length = config['max_length']
        self.token_to_id = config['token_to_id']
        self.id_to_token = config['id_to_token']
        self.pad_token = config['special_tokens']['pad']
        self.unk_token = config['special_tokens']['unk']
        self.bos_token = config['special_tokens']['bos']
        self.eos_token = config['special_tokens']['eos']
    
    @property
    def pad_token_id(self) -> int:
        return self.pad_token
    
    @property
    def unk_token_id(self) -> int:
        return self.unk_token
    
    @property
    def bos_token_id(self) -> int:
        return self.bos_token
    
    @property
    def eos_token_id(self) -> int:
        return self.eos_token

    def get_vocab_size(self) -> int:
        return self.vocab_size
