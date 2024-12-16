"""
ARNet Inference Module
Handles model saving, loading, and inference functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from arnets_old import ARNetLM, ResonanceModule, SimpleTokenizer
from metrics import ARNetMetrics

class ARNetInference:
    def __init__(self, model_dir: str = "research_results"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = None
        
    def save_model(self, model: nn.Module, tokenizer: SimpleTokenizer, config: Dict, 
                  metrics: Optional[ARNetMetrics] = None) -> None:
        """Save model, tokenizer, config, and metrics"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        tokenizer_data = {
            "word_to_ix": tokenizer.word_to_ix,
            "ix_to_word": {int(k): v for k, v in tokenizer.ix_to_word.items()},
            "max_vocab_size": tokenizer.max_vocab_size
        }
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_data, f)
        
        # Save config
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        # Save metrics if provided
        if metrics is not None:
            metrics_dict = metrics.calculate_metrics()
            metrics_path = os.path.join(self.model_dir, "training_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics_dict, f)
                
        print(f"Model and associated files saved to {self.model_dir}")
    
    def load_model(self) -> Tuple[nn.Module, SimpleTokenizer]:
        """Load model, tokenizer, and config"""
        # Load config
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Load tokenizer
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        with open(tokenizer_path, "r") as f:
            tokenizer_data = json.load(f)
        
        self.tokenizer = SimpleTokenizer(tokenizer_data["max_vocab_size"])
        self.tokenizer.word_to_ix = tokenizer_data["word_to_ix"]
        self.tokenizer.ix_to_word = {int(k): v for k, v in tokenizer_data["ix_to_word"].items()}
        
        # Initialize and load model
        self.model = ARNetLM(
            vocab_size=len(self.tokenizer.word_to_ix),
            embedding_dim=self.config["embedding_dim"],
            hidden_sizes=[self.config["hidden_size"]]
        )
        
        model_path = os.path.join(self.model_dir, "model.pt")
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        return self.model, self.tokenizer
    
    def generate_text(self, seed_text: str, max_len: int = 20, k: int = 5, 
                     temperature: float = 0.8, num_samples: int = 1) -> List[str]:
        """Generate text samples using the loaded model"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        generated_samples = []
        
        for _ in range(num_samples):
            input_ids = torch.tensor(self.tokenizer.encode(seed_text)).unsqueeze(0).to(self.device)
            generated_sequence = input_ids.tolist()[0]
            used_tokens = set()
            
            with torch.no_grad():
                for _ in range(max_len):
                    lengths = torch.tensor([input_ids.size(1)]).cpu().long()
                    predictions, _ = self.model(input_ids, lengths)
                    next_token_logits = predictions[-1][0, -1, :]
                    
                    # Penalize used tokens
                    for token in used_tokens:
                        next_token_logits[token] /= 2.0
                    
                    # Add noise for diversity
                    next_token_logits = torch.where(
                        next_token_logits < -1e4,
                        next_token_logits,
                        next_token_logits + torch.randn_like(next_token_logits) * 0.1
                    )
                    
                    # Apply temperature and get top k
                    logits = next_token_logits / temperature
                    top_k_logits, top_k_indices = torch.topk(logits, k)
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token_id = top_k_indices[torch.multinomial(probs, num_samples=1).item()].item()
                    
                    if next_token_id == self.tokenizer.word_to_ix["<PAD>"]:
                        break
                    
                    used_tokens.add(next_token_id)
                    generated_sequence.append(next_token_id)
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)
                    
                    if len(used_tokens) > 10:
                        used_tokens.clear()
            
            generated_text = self.tokenizer.decode(generated_sequence)
            generated_samples.append(generated_text)
        
        return generated_samples
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze text using the model's internal representations"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        self.model.eval()
        with torch.no_grad():
            # Tokenize and prepare input
            input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.device)
            lengths = torch.tensor([input_ids.size(1)]).cpu().long()
            
            # Get model outputs
            predictions, hidden_states = self.model(input_ids, lengths)
            
            # Analyze predictions
            probs = F.softmax(predictions[-1], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            
            # Analyze hidden states
            hidden_analysis = {
                f"layer_{i}": {
                    "mean": h.mean().item(),
                    "std": h.std().item(),
                    "max": h.max().item(),
                    "min": h.min().item(),
                    "sparsity": (h.abs() < 0.01).float().mean().item()
                }
                for i, h in enumerate(hidden_states)
            }
            
            return {
                "prediction_entropy": entropy,
                "hidden_state_analysis": hidden_analysis,
                "sequence_length": input_ids.size(1),
                "unique_tokens": len(set(input_ids.squeeze().tolist()))
            }

if __name__ == "__main__":
    # Example usage
    inference = ARNetInference()
    
    # Generate text samples
    samples = inference.generate_text(
        seed_text="The quick brown fox",
        max_len=30,
        num_samples=3
    )
    print("\nGenerated Samples:")
    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample}")
    
    # Analyze text
    analysis = inference.analyze_text("The quick brown fox jumps over the lazy dog")
    print("\nText Analysis:")
    print(json.dumps(analysis, indent=2))
