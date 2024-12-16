import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import sys
import os
import math

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.data import texts

class ResonanceModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.1, memory_size=32):
        super(ResonanceModule, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional_size = hidden_size * 2
        self.vocab_size = vocab_size
        
        # Embedding with learned positional encoding
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(512, hidden_size))
        
        # Bidirectional GRU with gradient checkpointing
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Enhanced attention mechanism
        self.self_attention = nn.MultiheadAttention(self.bidirectional_size, 4, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(self.bidirectional_size, 4, dropout=dropout)
        
        # Adaptive memory buffer with attention
        self.memory_buffer = nn.Parameter(torch.randn(memory_size, self.bidirectional_size))
        self.memory_attention = nn.MultiheadAttention(self.bidirectional_size, 4, dropout=dropout)
        self.memory_gate = nn.Sequential(
            nn.Linear(self.bidirectional_size * 2, self.bidirectional_size),
            nn.Sigmoid()
        )
        
        # Improved projection layers
        self.projection = nn.Sequential(
            nn.Linear(self.bidirectional_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Enhanced output layer with bottleneck
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )
        
        self.layer_norm1 = nn.LayerNorm(self.bidirectional_size)
        self.layer_norm2 = nn.LayerNorm(self.bidirectional_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Weight tying
        self.output_layer[-1].weight = nn.Parameter(self.embedding.weight.clone())
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
        
        # Special initialization for memory buffer
        nn.init.normal_(self.memory_buffer, mean=0.0, std=0.02)
    
    def forward(self, x, lengths, hidden=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Safety check for input indices
        x = torch.clamp(x, 0, self.vocab_size - 1)
        
        # Embedding with positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_enc = self.pos_encoder[:seq_len]
        embedded = self.embedding(x)
        embedded = self.dropout(embedded + pos_enc)
        
        # GRU with gradient checkpointing
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.gru(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        # Multi-stage attention
        # 1. Self-attention
        attn_output, _ = self.self_attention(
            output.transpose(0, 1),
            output.transpose(0, 1),
            output.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)
        output = self.layer_norm1(attn_output + output)
        
        # 2. Memory attention
        # Reshape memory buffer for attention
        memory = self.memory_buffer.unsqueeze(1)  # [memory_size, 1, hidden]
        memory = memory.expand(-1, batch_size, -1)  # [memory_size, batch, hidden]
        
        # Apply memory attention
        memory_output, _ = self.memory_attention(
            output.transpose(0, 1),  # [seq_len, batch, hidden]
            memory,  # [memory_size, batch, hidden]
            memory   # [memory_size, batch, hidden]
        )
        memory_output = memory_output.transpose(0, 1)
        
        # Adaptive memory gating
        gate = self.memory_gate(torch.cat([output, memory_output], dim=-1))
        output = output + gate * memory_output
        output = self.layer_norm2(output)
        
        # Update memory buffer with EMA and attention
        with torch.no_grad():
            # Calculate attention weights
            last_state = output[:, -1].unsqueeze(0)  # [1, batch, hidden]
            attention_weights = torch.matmul(self.memory_buffer, last_state.transpose(1, 2))
            attention_weights = F.softmax(attention_weights / math.sqrt(self.bidirectional_size), dim=0)
            
            # Update memory
            memory_update = torch.matmul(attention_weights, last_state)
            self.memory_buffer.data = 0.95 * self.memory_buffer.data + 0.05 * memory_update.squeeze(0)
        
        # Final processing
        output = self.projection(output)
        output = self.layer_norm3(output)
        prediction = self.output_layer(output)
        
        return prediction, output, hidden

class ARNetLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, num_layers=2, dropout=0.1):
        super(ARNetLM, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input layer
        self.input_layer = ResonanceModule(vocab_size, hidden_size, dropout)
        
        # Multiple processing layers
        self.layers = nn.ModuleList([
            ResonanceModule(vocab_size, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each layer output shape [batch_size, seq_len, vocab_size]
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([vocab_size])
            for _ in range(num_layers)
        ])
        
        # Expert mixing weights
        self.expert_weights = nn.ParameterList([
            nn.Parameter(torch.ones(num_layers))
            for _ in range(num_layers)
        ])
        
        # Dimension adjusters if needed
        self.dimension_adjusters = nn.ModuleList([
            nn.Linear(vocab_size, vocab_size)
            for _ in range(num_layers)
        ])
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        # Convert lengths to Long tensor
        lengths = lengths.long()
        
        # Initial layer
        prediction, layer_output, _ = self.input_layer(x, lengths)
        
        # Process through each layer
        for i, layer in enumerate(self.layers):
            # Get layer outputs
            layer_pred, layer_output, _ = layer(x, lengths)
            
            # Ensure dimensions match before adding
            if layer_pred.size() != prediction.size():
                # Adjust layer_output dimensions if needed
                if layer_pred.size(-1) != prediction.size(-1):
                    layer_pred = self.dimension_adjusters[i](layer_pred)
            
            # Apply expert mixing
            expert_weights = F.softmax(self.expert_weights[i], dim=0)
            layer_pred = layer_pred * expert_weights[i]
            
            # Residual connection
            prediction = prediction + layer_pred
            
            # Layer normalization over the vocabulary dimension
            prediction = self.layer_norms[i](prediction)
        
        return prediction, layer_output

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

def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    """Filter a distribution of logits using nucleus (top-p) filtering."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits

def generate_text(model, tokenizer, seed_text, max_len=30, min_len=10, top_k=50, top_p=0.9, 
                 temperature=0.7, repetition_penalty=1.2, device=None):
    """Generate text using both top-k and nucleus sampling with repetition penalty."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Initialize generation state
    tokens = torch.tensor(tokenizer.encode(seed_text)).unsqueeze(0).to(device)
    generated = tokens
    generated_tokens = tokens.tolist()[0]
    
    # Track n-gram frequencies for better repetition control
    ngram_counts = {2: {}, 3: {}, 4: {}}
    
    # Context window for coherence
    context_size = 5
    
    def get_ngrams(tokens, n):
        """Get n-grams from token sequence"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def update_ngram_counts(tokens):
        """Update n-gram frequency counts"""
        for n in ngram_counts:
            for ngram in get_ngrams(tokens, n):
                ngram_counts[n][ngram] = ngram_counts[n].get(ngram, 0) + 1
    
    def get_repetition_penalty(token, context):
        """Calculate dynamic repetition penalty based on n-gram frequencies"""
        penalty = repetition_penalty
        
        # Increase penalty for tokens that would create repeated n-grams
        for n in range(2, 5):
            if len(context) >= n-1:
                potential_ngram = tuple(context[-(n-1):] + [token])
                if potential_ngram[:-1] in ngram_counts[n]:
                    penalty *= (1 + 0.2 * ngram_counts[n][potential_ngram[:-1]])
        
        # Additional penalty for immediate repetition
        if context and token == context[-1]:
            penalty *= 1.5
        
        return penalty
    
    def adjust_temperature(context):
        """Dynamically adjust temperature based on generation state"""
        # Start conservative
        temp = temperature
        
        # Check for repetition patterns
        if len(context) >= 4:
            last_4 = context[-4:]
            unique_tokens = len(set(last_4))
            
            # Increase temperature if repetition detected
            if unique_tokens == 1:  # Same token repeated
                temp = min(1.5, temp * 1.3)
            elif unique_tokens == 2:  # Alternating tokens
                temp = min(1.3, temp * 1.2)
            else:
                # Cool down if diversity is good
                temp = max(0.6, temp * 0.95)
        
        return temp
    
    # Generate text
    with torch.no_grad():
        for i in range(max_len):
            # Get current context
            context = generated_tokens[-context_size:]
            
            # Forward pass
            seq_length = torch.tensor([generated.size(1)], device=device).long()
            predictions, _ = model(generated, seq_length)
            next_token_logits = predictions[0, -1, :].float()
            
            # Apply dynamic temperature
            current_temp = adjust_temperature(context)
            next_token_logits = next_token_logits / current_temp
            
            # Apply penalties and filtering
            for token_idx in range(len(next_token_logits)):
                penalty = get_repetition_penalty(token_idx, context)
                next_token_logits[token_idx] /= penalty
            
            # Filter with top-k
            top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits.fill_(-float('Inf'))
            next_token_logits[top_k_indices] = top_k_logits
            
            # Filter with nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(dim=0, index=sorted_indices, src=sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float('Inf')
            
            # Add small noise for diversity after minimum length
            if i >= min_len:
                noise = torch.randn_like(next_token_logits) * 0.02
                next_token_logits += noise
            
            # Sample token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Early stopping conditions
            if i >= min_len:
                if next_token.item() in [0, 1]:  # PAD or UNK
                    break
                
                # Check for excessive repetition
                if len(context) >= 4:
                    last_tokens = set(context[-4:])
                    if len(last_tokens) == 1 and next_token.item() in last_tokens:
                        break
            
            # Update state
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())
            update_ngram_counts(generated_tokens)
    
    # Clean up and return generated text
    generated_text = tokenizer.decode(generated[0].cpu().numpy())
    return generated_text

def train_arnet_lm(model, data_loader, num_epochs, lr=0.001, clip_value=1.0):
    """Train the ARNet language model"""
    print(f"\nTraining for {num_epochs} epochs...")
    
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)  # Add L2 regularization
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')  # Use per-token loss
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        token_count = 0
        
        try:
            for batch_idx, batch_data in enumerate(data_loader):
                try:
                    # Unpack batch data
                    if len(batch_data) == 3:
                        data, lengths, _ = batch_data
                    else:
                        data, lengths = batch_data
                    
                    # Move data to device
                    data = data.to(device)
                    lengths = lengths.to(device)
                    
                    # Forward pass
                    predictions, _ = model(data, lengths)
                    
                    # Prepare target (shift input by 1)
                    target = data[:, 1:].contiguous()
                    predictions = predictions[:, :-1].contiguous()
                    
                    # Calculate loss with token-level granularity
                    B, T, V = predictions.size()
                    loss = criterion(
                        predictions.view(-1, V),
                        target.view(-1)
                    )
                    
                    # Mask out padding tokens and calculate mean
                    mask = (target.view(-1) != 0).float()
                    loss = (loss * mask).sum() / mask.sum()
                    
                    # Add KL divergence loss to encourage diversity
                    token_probs = F.softmax(predictions.view(-1, V), dim=-1)
                    uniform_probs = torch.ones_like(token_probs) / V
                    kl_loss = F.kl_div(token_probs.log(), uniform_probs, reduction='batchmean')
                    loss = loss + 0.1 * kl_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    if clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            clip_value
                        )
                    
                    optimizer.step()
                    
                    total_loss += loss.item() * mask.sum().item()
                    token_count += mask.sum().item()
                    batch_count += 1
                    
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # Calculate average loss per token
            avg_loss = total_loss / token_count if token_count > 0 else 0
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:  # Stop if no improvement for 5 epochs
                    print("\nEarly stopping triggered!")
                    break
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            break
        except Exception as e:
            print(f"Error during epoch {epoch+1}: {str(e)}")
            continue

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

vocab_size = 1000  
hidden_size = 64  
num_layers = 2  
batch_size = 16  
num_epochs = 30  

model = ARNetLM(vocab_size, hidden_size, num_layers)

# Assuming you have a large list of texts
tokenizer = SimpleTokenizer(max_vocab_size=vocab_size)
tokenizer.fit(texts)
dataset = TextDataset(texts, tokenizer)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# First do fast learning
print("\nPerforming fast learning...")
fast_learn(model, data_loader, lr=0.001, num_steps=3)

# Then do regular training
print("\nStarting regular training...")
train_arnet_lm(model, data_loader, num_epochs, lr=0.001, clip_value=1.0)

# Generate samples with different parameters
print("\nGenerating samples:")
print("\nSeed: 'ARNet is'")
for _ in range(3):
    text = generate_text(
        model, tokenizer, "ARNet is",
        max_len=25, min_len=10,
        top_k=50, top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2
    )
    print(f"Generated: {text}")

print("\nSeed: 'The language model'")
for _ in range(3):
    text = generate_text(
        model, tokenizer, "The language model",
        max_len=30, min_len=15,
        top_k=40, top_p=0.85,
        temperature=0.8,
        repetition_penalty=1.3
    )
    print(f"Generated: {text}")

print("\nSeed: 'In the future'")
for _ in range(3):
    text = generate_text(
        model, tokenizer, "In the future",
        max_len=35, min_len=20,
        top_k=60, top_p=0.92,
        temperature=0.75,
        repetition_penalty=1.25
    )
    print(f"Generated: {text}")