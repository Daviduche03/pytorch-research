"""
ARNet Metrics and Analysis Tools
This module contains classes and functions for tracking, analyzing, and visualizing
ARNet model performance and behavior.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Union, Any
import time

def calculate_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate mutual information between two arrays"""
    # Compute joint probability distribution
    hist_2d, x_edges, y_edges = np.histogram2d(x.flatten(), y.flatten(), bins=20)
    p_xy = hist_2d / float(np.sum(hist_2d))
    
    # Compute marginal probability distributions
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    # Compute mutual information
    mutual_info = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i,j] > 0:
                mutual_info += p_xy[i,j] * np.log2(p_xy[i,j] / (p_x[i] * p_y[j]))
    
    return mutual_info

class ARNetMetrics:
    """Evaluation and analysis metrics for ARNet research"""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all tracking metrics"""
        self.loss_history = []
        self.val_loss_history = []
        self.resonance_patterns = []
        self.attention_patterns = []
        self.learning_rates = []
        self.gradient_norms = []
        self.activation_stats = defaultdict(list)
        self.memory_usage = []
        self.inference_times = []
        self.batch_times = []
        self.epochs = []
    
    def update(self, loss: float, batch_time: float, epoch: int):
        """Update training metrics"""
        self.loss_history.append(loss)
        self.batch_times.append(batch_time)
        self.epochs.append(epoch)
    
    def update_validation(self, val_loss: float):
        """Update validation metrics"""
        self.val_loss_history.append(val_loss)
    
    def update_training_metrics(self, loss: float, model: nn.Module, batch_size: int):
        """Track training metrics"""
        self.loss_history.append(loss)
        
        # Track gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.gradient_norms.append(np.sqrt(total_norm))
        
        # Track memory usage
        self.memory_usage.append({
            'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            'batch_size': batch_size
        })
    
    def track_resonance(self, resonance_values: torch.Tensor):
        """Track resonance patterns during training"""
        self.resonance_patterns.append(resonance_values.detach().cpu().numpy())
    
    def track_attention(self, attention_weights: torch.Tensor):
        """Track attention patterns"""
        self.attention_patterns.append(attention_weights.detach().cpu().numpy())
    
    def track_layer_activations(self, layer_name: str, activations: torch.Tensor):
        """Track statistics of layer activations"""
        acts = activations.detach().cpu().numpy()
        self.activation_stats[layer_name].append({
            'mean': np.mean(acts),
            'std': np.std(acts),
            'sparsity': np.mean(acts == 0),
            'max': np.max(acts),
            'min': np.min(acts)
        })
    
    def plot_training_curves(self, save_dir: str):
        """Plot and save training curves"""
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        plt.subplot(2, 1, 1)
        plt.plot(self.loss_history, label='Training Loss')
        if self.val_loss_history:
            plt.plot(self.val_loss_history, label='Validation Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot batch times
        plt.subplot(2, 1, 2)
        plt.plot(self.batch_times, label='Batch Time')
        plt.title('Batch Processing Time')
        plt.xlabel('Batch')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_plots.png'))
        plt.close()
    
    def calculate_metrics(self) -> Dict:
        """Calculate research metrics"""
        metrics = {
            'loss_convergence': {
                'final_loss': self.loss_history[-1] if self.loss_history else None,
                'convergence_rate': self._calculate_convergence_rate(),
                'stability': np.std(self.loss_history[-10:]) if len(self.loss_history) >= 10 else None
            },
            'resonance_analysis': {
                'mean_resonance': np.mean(self.resonance_patterns) if self.resonance_patterns else None,
                'resonance_stability': np.std(self.resonance_patterns) if self.resonance_patterns else None,
                'pattern_correlation': self._calculate_pattern_correlation()
            },
            'attention_analysis': {
                'attention_entropy': self._calculate_attention_entropy(),
                'attention_sparsity': self._calculate_attention_sparsity()
            },
            'efficiency_metrics': {
                'mean_inference_time': np.mean(self.inference_times) if self.inference_times else None,
                'memory_efficiency': self._calculate_memory_efficiency(),
                'parameter_utilization': self._calculate_parameter_utilization()
            }
        }
        return metrics
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate the rate of loss convergence"""
        if len(self.loss_history) < 2:
            return 0.0
        
        # Fit exponential decay to loss curve
        x = np.arange(len(self.loss_history))
        y = np.array(self.loss_history)
        
        # Simple ratio as fallback
        return (self.loss_history[0] - self.loss_history[-1]) / len(self.loss_history)
    
    def _calculate_pattern_correlation(self) -> float:
        """Calculate temporal correlation in resonance patterns"""
        if len(self.resonance_patterns) < 2:
            return 0.0
        
        patterns = np.array(self.resonance_patterns)
        correlations = np.corrcoef(patterns.reshape(patterns.shape[0], -1))
        return np.mean(correlations[np.triu_indices_from(correlations, k=1)])
    
    def _calculate_attention_entropy(self) -> float:
        """Calculate entropy of attention distributions"""
        if not self.attention_patterns:
            return 0.0
        
        attention = np.array(self.attention_patterns)
        entropy = -np.sum(attention * np.log(attention + 1e-10), axis=-1)
        return np.mean(entropy)
    
    def _calculate_attention_sparsity(self) -> float:
        """Calculate sparsity of attention patterns"""
        if not self.attention_patterns:
            return 0.0
        
        attention = np.array(self.attention_patterns)
        return np.mean(attention < 0.01)
    
    def _calculate_memory_efficiency(self) -> Dict:
        """Calculate memory usage efficiency"""
        if not self.memory_usage:
            return {}
            
        usage = np.array([m['allocated'] for m in self.memory_usage])
        batch_sizes = np.array([m['batch_size'] for m in self.memory_usage])
        
        return {
            'memory_per_sample': np.mean(usage / batch_sizes),
            'peak_memory': np.max(usage),
            'memory_stability': np.std(usage / batch_sizes)
        }
    
    def _calculate_parameter_utilization(self) -> Dict:
        """Calculate parameter utilization metrics"""
        if not self.gradient_norms:
            return {}
        
        return {
            'mean_gradient_norm': np.mean(self.gradient_norms),
            'gradient_stability': np.std(self.gradient_norms),
            'zero_gradient_ratio': np.mean(np.array(self.gradient_norms) < 1e-6)
        }

class ARNetResearchAnalysis:
    """Research analysis tools for ARNet"""
    
    def __init__(self, model: nn.Module, metrics: ARNetMetrics):
        self.model = model
        self.metrics = metrics
    
    def analyze_model_behavior(self, input_data: torch.Tensor) -> Dict:
        """Analyze model's behavior on specific input"""
        self.model.eval()
        with torch.no_grad():
            start_time = time.time()
            predictions, hidden_states = self.model(input_data, torch.tensor([input_data.size(1)]))
            inference_time = time.time() - start_time
            
            # Get the final layer predictions
            final_output = predictions[-1]
            
            analysis = {
                'inference_time': inference_time,
                'output_statistics': {
                    'mean': final_output.mean().item(),
                    'std': final_output.std().item(),
                    'max': final_output.max().item(),
                    'min': final_output.min().item()
                },
                'hidden_state_statistics': {
                    'mean': torch.mean(torch.stack([h.mean() for h in hidden_states])).item(),
                    'std': torch.mean(torch.stack([h.std() for h in hidden_states])).item()
                }
            }
            
            return analysis
    
    def analyze_learning_dynamics(self) -> Dict:
        """Analyze learning dynamics from collected metrics"""
        return {
            'convergence_speed': self._analyze_convergence(),
            'stability_metrics': self._analyze_stability(),
            'efficiency_metrics': self._analyze_efficiency()
        }
    
    def _analyze_convergence(self) -> Dict:
        """Analyze convergence characteristics"""
        loss_history = np.array(self.metrics.loss_history)
        
        return {
            'iterations_to_converge': self._estimate_convergence_point(loss_history),
            'convergence_rate': self._calculate_convergence_rate(loss_history),
            'final_loss': loss_history[-1] if len(loss_history) > 0 else None
        }
    
    def _calculate_convergence_rate(self, loss_history: np.ndarray) -> float:
        """Calculate the rate of loss convergence"""
        if len(loss_history) < 2:
            return 0.0
        
        # Calculate the average rate of loss decrease per iteration
        total_decrease = loss_history[0] - loss_history[-1]
        num_iterations = len(loss_history) - 1
        
        if num_iterations == 0:
            return 0.0
            
        # Add small epsilon to avoid division by zero
        rate = total_decrease / (num_iterations + 1e-10)
        
        # Normalize by initial loss to get relative convergence rate
        if loss_history[0] != 0:
            rate = rate / abs(loss_history[0])
            
        return rate
    
    def _estimate_convergence_point(self, loss_history: np.ndarray) -> int:
        """Estimate the point of convergence"""
        if len(loss_history) < 2:
            return 0
            
        # Use sliding window to detect stability
        window_size = min(10, len(loss_history) // 4)
        std_window = np.std([
            np.std(loss_history[i:i+window_size])
            for i in range(0, len(loss_history) - window_size)
        ])
        
        # Find point where variation stabilizes
        for i in range(window_size, len(loss_history) - window_size):
            if np.std(loss_history[i:i+window_size]) < std_window * 0.1:
                return i
        
        return len(loss_history)
    
    def _analyze_stability(self) -> Dict:
        """Analyze training stability"""
        return {
            'loss_stability': np.std(self.metrics.loss_history[-10:]) if len(self.metrics.loss_history) >= 10 else None,
            'gradient_stability': np.std(self.metrics.gradient_norms) if self.metrics.gradient_norms else None,
            'memory_stability': self._analyze_memory_stability()
        }
    
    def _analyze_memory_stability(self) -> Dict:
        """Analyze memory usage stability"""
        if not self.metrics.memory_usage:
            return {}
            
        memory = np.array([m['allocated'] for m in self.metrics.memory_usage])
        return {
            'mean_usage': np.mean(memory),
            'std_usage': np.std(memory),
            'peak_usage': np.max(memory)
        }
    
    def _analyze_efficiency(self) -> Dict:
        """Analyze computational efficiency"""
        return {
            'memory_efficiency': self._calculate_memory_efficiency(),
            'computational_efficiency': self._calculate_computational_efficiency()
        }
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        if not self.metrics.memory_usage or not self.metrics.loss_history:
            return 0.0
            
        # Calculate efficiency as performance improvement per memory used
        memory = np.array([m['allocated'] for m in self.metrics.memory_usage])
        loss_improvement = self.metrics.loss_history[0] - self.metrics.loss_history[-1]
        
        return loss_improvement / (np.mean(memory) + 1e-10)
    
    def _calculate_computational_efficiency(self) -> Dict:
        """Calculate computational efficiency metrics"""
        return {
            'mean_inference_time': np.mean(self.metrics.inference_times) if self.metrics.inference_times else None,
            'inference_time_stability': np.std(self.metrics.inference_times) if self.metrics.inference_times else None
        }
