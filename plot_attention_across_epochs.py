#!/usr/bin/env python3
"""
Script to plot average unnormalized attention to root edges across different training epochs.
Analyzes how unnormalized attention patterns (raw Q@K^T scores) evolve during training.
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoConfig
from coconut import Coconut
from stokenizer import STokenizer
from tqdm import tqdm
import os
from analyze_attention import get_attention_first_step, get_reachable_nodes_at_step


def load_model_for_epoch(epoch, checkpoint_dir="ckpts/coconut-run-aug18"):
    """Load the Coconut model for a specific epoch."""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model for epoch {epoch}...")
    
    # Load weights
    saved_weights = torch.load(
        checkpoint_path, 
        map_location=torch.device("cuda:0")
    )
    
    # Initialize tokenizer and model
    tokenizer = STokenizer()
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    model = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained("configs/symbol-2layer-8head-768dim.json")
    )
    model = Coconut(model, latent_id, start_latent_id, end_latent_id, tokenizer.eos_token_id)
    model.load_state_dict(saved_weights, strict=False)
    model.eval()
    model.base_causallm.to("cuda:0")
    
    return model, tokenizer


def apply_unnormalized_attention_patch():
    """
    Apply monkey patch to return unnormalized attention weights instead of normalized ones.
    """
    from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward
    
    # Store original function
    original_eager_attention = eager_attention_forward
    
    def patched_eager_attention(module, query, key, value, attention_mask, head_mask=None, **kwargs):
        # Compute raw attention weights (Q @ K^T)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        # Apply scaling if enabled (this is still part of "unnormalized" before softmax)
        if module.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        
        # Apply layer-wise attention scaling if enabled
        if module.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(module.layer_idx + 1)
        
        # For the actual model computation, we still need to do the full attention
        # So call the original function to get the proper output
        attn_output, _ = original_eager_attention(module, query, key, value, attention_mask, head_mask, **kwargs)
        
        # Return the output but with unnormalized (pre-softmax) weights
        return attn_output, attn_weights
    
    # Monkey patch the function
    import transformers.models.gpt2.modeling_gpt2
    transformers.models.gpt2.modeling_gpt2.eager_attention_forward = patched_eager_attention
    
    return original_eager_attention


def restore_attention_patch(original_function):
    """Restore the original eager_attention_forward function."""
    import transformers.models.gpt2.modeling_gpt2
    transformers.models.gpt2.modeling_gpt2.eager_attention_forward = original_function



def analyze_sample_attention(model, tokenizer, data_entry, step=0):
    """Analyze unnormalized attention for a single sample at a specific reasoning step."""
    # Construct question with appropriate number of latent tokens for the step
    edges_str = "|".join([f" {e[0]} {e[1]} " for e in data_entry['edges']]).strip()
    latent_tokens = " <|latent|>" * step  # step latent tokens for step i
    question = f"<eos> {edges_str} [Q] {data_entry['target']} {data_entry['neg_target']} [R] {data_entry['root']}{latent_tokens}"
    
    try:
        # Apply monkey patch to get unnormalized attention weights
        original_function = apply_unnormalized_attention_patch()
        
        try:
            second_layer_attention = get_attention_first_step(model, tokenizer, question, layer=1)
        finally:
            # Always restore the original function
            restore_attention_patch(original_function)
            
    except Exception as e:
        return None, None
    
    # Sum attention across all heads for the last position (latent token position)
    attention_weights = second_layer_attention.sum(dim=0)[-1, :]
    
    # Get nodes that are reachable at this step
    frontier_nodes, all_reachable_nodes = get_reachable_nodes_at_step(data_entry, step)
    
    # Parse the question to find edge token positions
    tokens = question.split(" ")
    
    frontier_edge_attentions = []
    other_edge_attentions = []
    
    # Find edges and their corresponding <e> token positions
    i = 1  # Start after <eos>
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i + 2] == "|":
            start_node = int(tokens[i])
            edge_token_pos = i + 2
            
            if edge_token_pos < len(attention_weights):
                attention_value = attention_weights[edge_token_pos].item()
                
                if start_node in frontier_nodes:
                    frontier_edge_attentions.append(attention_value)
                else:
                    other_edge_attentions.append(attention_value)
            
            i += 3
        else:
            break
    
    # Calculate averages
    avg_frontier = np.mean(frontier_edge_attentions) if len(frontier_edge_attentions) > 0 else None
    avg_other = np.mean(other_edge_attentions) if len(other_edge_attentions) > 0 else None
        
    return avg_frontier, avg_other

def analyze_epoch(epoch, dataset, steps, checkpoint_dir="ckpts/bfs-run-aug11", max_samples=None):
    """Analyze unnormalized attention patterns for a specific epoch across multiple reasoning steps."""
    try:
        model, tokenizer = load_model_for_epoch(epoch, checkpoint_dir)
    except FileNotFoundError as e:
        print(f"Skipping epoch {epoch}: {e}")
        return None
    
    if max_samples:
        dataset_subset = dataset[:max_samples]
    else:
        dataset_subset = dataset
    
    results = {}
    
    for step in steps:
        frontier_attention_scores = []
        other_attention_scores = []
        processed_count = 0
        
        for data_entry in tqdm(dataset_subset, desc=f"Epoch {epoch}, Step {step}", leave=False):
            avg_frontier, avg_other = analyze_sample_attention(model, tokenizer, data_entry, step)
            
            if avg_frontier is not None and avg_other is not None:
                frontier_attention_scores.append(avg_frontier)
                other_attention_scores.append(avg_other)
                processed_count += 1
        
        # Calculate averages
        mean_frontier = np.mean(frontier_attention_scores) if frontier_attention_scores else 0.0
        mean_other = np.mean(other_attention_scores) if other_attention_scores else 0.0
        
        results[step] = {
            'mean_frontier_attention': mean_frontier,
            'mean_other_attention': mean_other,
            'processed_count': processed_count
        }
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot unnormalized attention evolution across training epochs")
    parser.add_argument("--test_file", default="data/prosqa_test_graph_4_coconut_shuffled.json", 
                       help="Path to test file")
    
    # ckpts/bfs-run-aug11
    parser.add_argument("--checkpoint_dir", default="ckpts/coconut-run-aug18", # default="ckpts/bfs-run-aug11", # default="ckpts/bfs-uniform-loss-sep5",
                       help="Directory containing checkpoints")
    parser.add_argument("--epochs", nargs="+", type=int, default=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350],# default=[10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225],
                       help="List of epochs to analyze")
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 1, 2, 3],
                       help="List of reasoning steps to analyze")
    parser.add_argument("--max_samples", type=int, default=200,
                       help="Maximum number of samples to process per epoch")
    # attention_evolution.png
    parser.add_argument("--output", default="unnormalized_attention_evolution_coconut.png",
                       help="Output plot filename")
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_file}")
    with open(args.test_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"Analyzing unnormalized attention across epochs: {args.epochs}")
    print(f"Analyzing reasoning steps: {args.steps}")
    print(f"Using max {args.max_samples} samples per epoch")
    
    # Analyze each epoch
    all_results = {}
    epochs_analyzed = []
    
    for epoch in args.epochs:
        epoch_results = analyze_epoch(
            epoch, dataset, args.steps, args.checkpoint_dir, args.max_samples
        )
        
        if epoch_results is not None:
            all_results[epoch] = epoch_results
            epochs_analyzed.append(epoch)
            print(f"Epoch {epoch}:")
            for step in args.steps:
                step_data = epoch_results[step]
                print(f"  Step {step}: Frontier={step_data['mean_frontier_attention']:.4f}, "
                      f"Other={step_data['mean_other_attention']:.4f}, "
                      f"Samples={step_data['processed_count']}")
    
    if not all_results:
        print("No epochs could be analyzed!")
        return
    
    # Save raw data
    raw_data = {
        'all_results': all_results,
        'epochs_analyzed': epochs_analyzed,
        'steps': args.steps,
        'checkpoint_dir': args.checkpoint_dir,
        'test_file': args.test_file,
        'max_samples': args.max_samples,
        'analysis_metadata': {
            'total_epochs': len(epochs_analyzed),
            'total_steps': len(args.steps),
            'generated_by': 'plot_attention_across_epochs.py'
        }
    }
    
    # Create raw data filename based on output filename
    raw_data_filename = args.output.replace('.png', '_raw_data.json')
    print(f"Saving raw data to {raw_data_filename}")
    with open(raw_data_filename, 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"Raw data saved successfully!")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors for different steps
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
    
    epochs = epochs_analyzed
    
    # Plot each step with different colors and line styles
    for i, step in enumerate(args.steps):
        color = colors[i % len(colors)]
        
        # Extract data for this step
        frontier_attentions = []
        other_attentions = []
        
        for epoch in epochs:
            step_data = all_results[epoch][step]
            frontier_attentions.append(step_data['mean_frontier_attention'])
            other_attentions.append(step_data['mean_other_attention'])
        
        # Plot frontier edges (solid line)
        plt.plot(epochs, frontier_attentions, 'o-', linewidth=2, markersize=8, 
                 label=f'Step {step} - Frontier edges', color=color)
        
        # Plot other edges (dotted line)
        plt.plot(epochs, other_attentions, 'o:', linewidth=2, markersize=6, 
                 label=f'Step {step} - Other edges', color=color, alpha=0.7)
    
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Average Unnormalized Attention Score', fontsize=12)
    plt.title('Evolution of Unnormalized Attention to Edge Tokens During Training\n(Solid lines: Frontier edges, Dotted lines: Other edges)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to only show analyzed epochs
    plt.xticks(epochs)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {args.output}")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Epochs analyzed: {epochs_analyzed}")
    print(f"Steps analyzed: {args.steps}")
    
    for step in args.steps:
        frontier_attentions = [all_results[epoch][step]['mean_frontier_attention'] for epoch in epochs]
        other_attentions = [all_results[epoch][step]['mean_other_attention'] for epoch in epochs]
        
        if len(frontier_attentions) >= 2:
            frontier_trend = "increasing" if frontier_attentions[-1] > frontier_attentions[0] else "decreasing"
            other_trend = "increasing" if other_attentions[-1] > other_attentions[0] else "decreasing"
            
            print(f"\nStep {step}:")
            print(f"  Frontier edge unnormalized attention trend: {frontier_trend} ({frontier_attentions[0]:.4f} → {frontier_attentions[-1]:.4f})")
            print(f"  Other edge unnormalized attention trend: {other_trend} ({other_attentions[0]:.4f} → {other_attentions[-1]:.4f})")
            
            # Calculate attention ratio evolution
            ratios = [f/o if o > 0 else 0 for f, o in zip(frontier_attentions, other_attentions)]
            if ratios:
                print(f"  Unnormalized attention ratio (frontier/other) evolution: {ratios[0]:.2f}x → {ratios[-1]:.2f}x")
    
    plt.show()


if __name__ == "__main__":
    main()
