
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.data import texts

class ResonanceModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_heads=2, dropout=0.2):
        super(ResonanceModule, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional_size = hidden_size * 2
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.self_attention = nn.MultiheadAttention(self.bidirectional_size, num_heads, dropout=dropout)
        self.memory_buffer = nn.Parameter(torch.randn(5, self.bidirectional_size))
        
        # Projection layer to reduce dimensionality after bidirectional GRU
        self.projection = nn.Linear(self.bidirectional_size, hidden_size)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        
        self.layer_norm1 = nn.LayerNorm(self.bidirectional_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Weight tying between embedding and output layer
        self.output_layer[-1].weight = nn.Parameter(self.embedding.weight.clone())
        
        # Initialize parameters with smaller values
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, x, lengths, hidden=None):
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.dropout(self.embedding(x))
        
        # GRU
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.gru(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        # Self-attention
        attn_output, _ = self.self_attention(
            output.transpose(0, 1),
            output.transpose(0, 1),
            output.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)
        
        # First residual connection and layer norm
        output = self.layer_norm1(attn_output + output)
        
        # Update memory buffer with exponential moving average
        with torch.no_grad():
            self.memory_buffer.data = 0.9 * self.memory_buffer.data + 0.1 * output[:, -1, :].mean(0, keepdim=True)
        
        # Add memory buffer to the output
        output = output + self.memory_buffer.mean(0, keepdim=True).expand_as(output)
        
        # Project down to hidden_size
        output = self.projection(output)
        output = self.layer_norm2(output)
        
        # Final output layer
        prediction = self.output_layer(output)
        
        return prediction, output, hidden

class ARNetLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_sizes):
        super(ARNetLM, self).__init__()
        # Reduce number of layers for small datasets
        self.layers = nn.ModuleList([ResonanceModule(vocab_size, embedding_dim, hidden_sizes[0])])
        
        # L2 regularization
        self.l2_reg = 1e-5
        
    def forward(self, x, lengths):
        hidden_states = []
        predictions = []
        for layer in self.layers:
            prediction, hidden, _ = layer(x, lengths)
            hidden_states.append(hidden)
            predictions.append(prediction)
            x = prediction.argmax(dim=-1)
            
        return predictions, hidden_states
    
    def get_reg_loss(self):
        """Calculate L2 regularization loss"""
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.norm(param, p=2)
        return self.l2_reg * reg_loss

class SimpleTokenizer:
    def __init__(self, max_vocab_size=10000):
        self.word_to_ix = {"<PAD>": 0, "<UNK>": 1}
        self.ix_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.max_vocab_size = max_vocab_size

    def fit(self, texts):
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.max_vocab_size-2]:  # -2 for <PAD> and <UNK>
            if word not in self.word_to_ix:
                self.word_to_ix[word] = len(self.word_to_ix)
                self.ix_to_word[len(self.ix_to_word)] = word

    def encode(self, text):
        return [self.word_to_ix.get(word, self.word_to_ix["<UNK>"]) for word in text.split()]

    def decode(self, indices):
        return " ".join([self.ix_to_word.get(ix, "<UNK>") for ix in indices])

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.data = [torch.tensor(tokenizer.encode(text)) for text in texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # Sort the batch by sequence length (descending order)
    sorted_batch = sorted(batch, key=lambda x: len(x), reverse=True)
    sequences = [x for x in sorted_batch]
    lengths = torch.tensor([len(x) for x in sequences])
    
    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Create targets (shifted by one position)
    targets = torch.zeros_like(padded)
    targets[:, :-1] = padded[:, 1:]
    
    return padded, lengths, targets

def top_k_sampling(logits, k=10, temperature=1.0):
    # Apply temperature
    logits = logits / max(temperature, 1e-6)
    
    # Get top k logits and corresponding indices
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # Apply softmax to get probabilities
    probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample from the distribution
    idx = torch.multinomial(probs, num_samples=1).item()
    
    return top_k_indices[idx].item()

def generate_text_top_k(model, tokenizer, seed_text, max_len=20, k=5, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize the seed text
    input_ids = torch.tensor(tokenizer.encode(seed_text)).unsqueeze(0).to(device)
    generated_sequence = input_ids.tolist()[0]
    
    # Track used tokens to prevent repetition
    used_tokens = set()
    
    with torch.no_grad():
        for _ in range(max_len):
            lengths = torch.tensor([input_ids.size(1)]).cpu().long()
            predictions, _ = model(input_ids, lengths)
            next_token_logits = predictions[-1][0, -1, :]
            
            # Penalize already used tokens
            for token in used_tokens:
                next_token_logits[token] /= 2.0
            
            # Prevent common tokens from dominating
            next_token_logits = torch.where(
                next_token_logits < -1e4,
                next_token_logits,
                next_token_logits + torch.randn_like(next_token_logits) * 0.1
            )
            
            next_token_id = top_k_sampling(next_token_logits, k=k, temperature=temperature)
            
            # Stop if we predict a padding token
            if next_token_id == tokenizer.word_to_ix["<PAD>"]:
                break
                
            used_tokens.add(next_token_id)
            generated_sequence.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
            
            # Reset used_tokens after a certain length to prevent too much restriction
            if len(used_tokens) > 10:
                used_tokens.clear()
    
    return tokenizer.decode(generated_sequence)

def fast_learn(model, data_loader, lr=0.001, num_steps=3):  
    """
    Implements rapid Hebbian-inspired learning with stability controls
    """
    device = next(model.parameters()).device
    model.eval()
    
    def update_weight_matrix(weight, lr_factor):
        """Helper function to update weight matrices with proper dimension handling"""
        out_dim, in_dim = weight.shape
        
        # Compute weight statistics for stability
        weight_mean = weight.mean()
        weight_std = torch.std(weight)
        
        # Compute correlation-based update
        if out_dim < in_dim:
            update = torch.mm(weight - weight_mean, (weight - weight_mean).t())
        else:
            update = torch.mm((weight - weight_mean).t(), weight - weight_mean)
        
        update_size = min(update.size(0), weight.size(0))
        
        # Create stabilized update
        weight_update = torch.zeros_like(weight)
        if out_dim < in_dim:
            weight_update[:update_size, :] = lr_factor * update[:update_size, :update_size].mm(weight[:update_size, :])
        else:
            weight_update[:, :update_size] = lr_factor * weight[:, :update_size].mm(update[:update_size, :update_size])
        
        # Apply conservative update
        max_update = 0.01 * weight_std  
        weight_update = torch.clamp(weight_update, -max_update, max_update)
        
        # Ensure update preserves weight statistics
        updated_weight = weight + F.normalize(weight_update, p=2, dim=1)
        if torch.std(updated_weight) > 1.5 * weight_std:
            weight_update *= 0.5  # Reduce update if it changes statistics too much
            
        return weight_update
    
    # Track original weight statistics
    orig_stats = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            orig_stats[name] = {
                'mean': param.data.mean().item(),
                'std': torch.std(param.data).item()
            }
    
    with torch.no_grad():
        # Get multiple batches for better statistics
        batch_data = []
        for i, (data, lens, _) in enumerate(data_loader):
            if i >= 3:  
                break
            batch_data.append((data.to(device), lens))
        
        for step in range(num_steps):
            lr_factor = lr / (step + 1)**2  
            
            for layer in model.layers:
                # Process each batch
                for data, lengths in batch_data:
                    # Forward pass to get activations
                    embedded = layer.embedding(data)
                    packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
                    gru_out, _ = layer.gru(packed)
                    gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
                    
                    # Update GRU weights
                    for direction in ['l0', 'l0_reverse'] if layer.gru.bidirectional else ['l0']:
                        for weight_type in ['weight_ih_', 'weight_hh_']:
                            weight = getattr(layer.gru, weight_type + direction)
                            hidden_size = layer.gru.hidden_size
                            
                            for gate_idx in range(3):
                                start_idx = gate_idx * hidden_size
                                end_idx = (gate_idx + 1) * hidden_size
                                gate = weight[start_idx:end_idx]
                                
                                weight_update = update_weight_matrix(gate, lr_factor)
                                weight.data[start_idx:end_idx] += weight_update
                    
                    # Update embedding weights
                    emb_weight = layer.embedding.weight
                    vocab_size, emb_dim = emb_weight.shape
                    
                    # Process valid embeddings
                    flat_embedded = embedded.view(-1, emb_dim)
                    valid_indices = (flat_embedded.sum(dim=1) != 0)
                    valid_embedded = flat_embedded[valid_indices]
                    
                    if len(valid_embedded) > 0:
                        emb_mean = valid_embedded.mean(0, keepdim=True)
                        emb_centered = valid_embedded - emb_mean
                        emb_corr = torch.mm(emb_centered.t(), emb_centered)
                        
                        update_size = min(emb_dim, vocab_size)
                        emb_update = torch.zeros_like(emb_weight)
                        emb_update[:update_size, :update_size] = lr_factor * 0.1 * emb_corr[:update_size, :update_size]  
                        
                        max_emb_update = 0.01 * torch.std(emb_weight)  
                        emb_update = torch.clamp(emb_update, -max_emb_update, max_emb_update)
                        emb_weight.data += F.normalize(emb_update, p=2, dim=1)
                    
                    # Update attention weights conservatively
                    if hasattr(layer, 'self_attention'):
                        for name, param in layer.self_attention.named_parameters():
                            if 'weight' in name:
                                weight_update = update_weight_matrix(param, lr_factor * 0.1)  
                                param.data += weight_update
                    
                    # Update other layers conservatively
                    for module in [layer.projection, *layer.output_layer]:
                        if isinstance(module, nn.Linear):
                            weight = module.weight
                            weight_update = update_weight_matrix(weight, lr_factor * 0.1)
                            module.weight.data += weight_update
            
            # Verify weight statistics haven't changed too much
            for name, param in model.named_parameters():
                if 'weight' in name and name in orig_stats:
                    curr_std = torch.std(param.data).item()
                    orig_std = orig_stats[name]['std']
                    if curr_std > 2 * orig_std:
                        # Reset if statistics changed too much
                        param.data *= orig_std / curr_std
            
            print(f"Fast learning step {step+1}/{num_steps} completed")

def train_arnet_lm(model, data_loader, num_epochs, lr=0.001, clip_value=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, epochs=num_epochs,
        steps_per_epoch=len(data_loader),
        pct_start=0.3, anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Apply fast learning only in first few epochs and if loss is decreasing
        if epoch < 3 and (epoch == 0 or total_loss < best_loss):
            print(f"\nApplying fast learning at epoch {epoch+1}")
            fast_learn(model, data_loader, lr=0.001, num_steps=3)  
            print("Fast learning completed\n")
        
        for batch_idx, (data, lengths, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            
            predictions, _ = model(data, lengths)
            loss = sum(criterion(pred.view(-1, pred.size(-1)), targets.view(-1)) for pred in predictions)
            loss += 0.01 * model.get_reg_loss()  
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Early stopping with more patience
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        if torch.isnan(torch.tensor(avg_loss)):
            print("NaN loss detected. Stopping training.")
            break

def generate_samples(model, tokenizer, seed_text, num_samples=5, max_len=20, k=10, temperature=0.3):
    print(f"Seed text: '{seed_text}'")
    print(f"Generating {num_samples} samples:")
    for i in range(num_samples):
        generated_text = generate_text_top_k(model, tokenizer, seed_text, max_len=max_len, k=k, temperature=temperature)
        print(f"Sample {i+1}: {generated_text}")

# Example usage
vocab_size = 1000  
embedding_dim = 64  
hidden_sizes = [64]  
batch_size = 16  
num_epochs = 30  

model = ARNetLM(vocab_size, embedding_dim, hidden_sizes)

# Assuming you have a large list of texts
tokenizer = SimpleTokenizer(max_vocab_size=vocab_size)
tokenizer.fit(texts)
dataset = TextDataset(texts, tokenizer)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

train_arnet_lm(model, data_loader, num_epochs, lr=0.0001, clip_value=0.5)

# Generate samples with different parameters
generate_samples(model, tokenizer, "ARNet is", num_samples=3, max_len=20, k=10, temperature=0.7)
generate_samples(model, tokenizer, "The language model", num_samples=3, max_len=25, k=5, temperature=1.0)
generate_samples(model, tokenizer, "In the future", num_samples=3, max_len=30, k=15, temperature=0.7)