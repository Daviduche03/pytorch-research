"""
ARNet Experiment Runner
This module contains functions for running and analyzing ARNet experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader
from metrics import ARNetMetrics, ARNetResearchAnalysis
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fast_learn(model: nn.Module, data_loader: DataLoader, lr: float = 0.001, num_steps: int = 3):  
    """
    Implements rapid Hebbian-inspired learning with stability controls
    """
    device = next(model.parameters()).device
    
    # Track original weight statistics
    orig_stats = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            orig_stats[name] = {
                'mean': param.data.mean().item(),
                'std': torch.std(param.data).item()
            }
    
    def update_weight_matrix(weight: torch.Tensor, lr_factor: float):
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
                    
                    # Temporarily set to train mode for GRU forward pass
                    layer.gru.train()
                    packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
                    gru_out, _ = layer.gru(packed)
                    gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
                    layer.gru.eval()  # Set back to eval mode
                    
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

def run_research_experiment(
    model: nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    save_dir: str = "research_results"
) -> Tuple[Any, Dict]:
    """Run a complete research experiment with comprehensive analysis"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize metrics tracking
    metrics = ARNetMetrics()
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Ensure model is in training mode for the entire epoch
        epoch_loss = 0.0
        batch_times = []
        
        for batch_idx, batch in enumerate(train_data):
            # Unpack the batch data
            data, lengths, targets = batch
            data, targets = data.to(device), targets.to(device)
            
            # Record batch start time
            batch_start = time.time()
            
            # Standard gradient-based learning
            optimizer.zero_grad()
            predictions, hidden_states = model(data, lengths)
            loss = F.cross_entropy(predictions[-1].view(-1, predictions[-1].size(-1)), targets.view(-1))
            
            # Add L2 regularization
            l2_loss = model.get_reg_loss()
            total_loss = loss + l2_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Apply fast learning every 10 batches
            if batch_idx % 10 == 0:
                # Store training mode state
                training_states = {}
                for name, module in model.named_modules():
                    training_states[name] = module.training
                
                # Apply fast learning
                fast_learn(model, train_data, lr=learning_rate * 0.1, num_steps=3)
                
                # Restore training mode state
                for name, module in model.named_modules():
                    if name in training_states:
                        module.train(training_states[name])
                
                # Ensure model is back in training mode
                model.train()
            
            # Record metrics
            epoch_loss += total_loss.item()
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_data)} | "
                      f"Loss: {total_loss.item():.4f} | Batch Time: {batch_time:.3f}s")
        
        # Compute epoch metrics
        avg_loss = epoch_loss / len(train_data)
        metrics.update(avg_loss, np.mean(batch_times), epoch)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Batch Time: {np.mean(batch_times):.3f}s")
        
        # Validation
        if val_data is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_data:
                    data, lengths, targets = batch
                    data, targets = data.to(device), targets.to(device)
                    predictions, _ = model(data, lengths)
                    val_loss += F.cross_entropy(predictions[-1].view(-1, predictions[-1].size(-1)), 
                                             targets.view(-1)).item()
            
            val_loss /= len(val_data)
            print(f"Validation Loss: {val_loss:.4f}")
            metrics.update_validation(val_loss)
    
    # Save training plots
    metrics.plot_training_curves(save_dir)
    
    # Analyze model behavior
    analyzer = ARNetResearchAnalysis(model, metrics)
    final_analysis = {}
    
    # Use validation data for behavior analysis if available
    if val_data is not None:
        try:
            # Get first batch from validation data
            val_batch = next(iter(val_data))
            data, lengths, _ = val_batch
            data = data.to(device)
            
            # Analyze model behavior on validation data
            final_analysis = analyzer.analyze_model_behavior(data)
        except Exception as e:
            print(f"Warning: Could not perform model behavior analysis: {str(e)}")
    
    # Analyze learning dynamics
    learning_analysis = analyzer.analyze_learning_dynamics()
    final_analysis.update(learning_analysis)
    
    # Save metrics to JSON
    metrics_dict = metrics.calculate_metrics()
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    # Save model and configuration
    config = {
        'embedding_dim': model.layers[0].embedding.embedding_dim,
        'hidden_size': model.layers[0].hidden_size,
        'num_heads': model.layers[0].self_attention.num_heads,
        'dropout': model.layers[0].dropout.p,
        'l2_reg': model.l2_reg
    }
    
    from inference import ARNetInference
    inference_handler = ARNetInference(save_dir)
    inference_handler.save_model(model, train_data.dataset.tokenizer, config, metrics)
    
    return metrics, final_analysis
