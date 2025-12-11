"""
Regenerate all 11 comprehensive visualizations from existing results.
"""
import sys
import os
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.comprehensive_visualization import create_all_visualizations
from src.utils.visualization import plot_comparison
from src.utils.metrics import compute_convergence_metrics


def regenerate_all():
    """Regenerate all visualizations from existing results."""
    print("=" * 60)
    print("Regenerating All Comprehensive Visualizations")
    print("=" * 60)
    
    # Q-Learning
    print("\n" + "-" * 60)
    print("Regenerating Q-Learning visualizations...")
    print("-" * 60)
    
    ql_results_path = "experiments/results/q_learning/results.json"
    if os.path.exists(ql_results_path):
        with open(ql_results_path, 'r') as f:
            ql_data = json.load(f)
        
        create_all_visualizations(
            ql_data['training_stats'],
            ql_data['evaluation'],
            ql_data.get('convergence', {}),
            "experiments/results/q_learning/plots",
            "Q-Learning"
        )
        print("✓ Q-Learning visualizations complete")
    else:
        print("✗ Q-Learning results not found")
    
    # PPO
    print("\n" + "-" * 60)
    print("Regenerating PPO visualizations...")
    print("-" * 60)
    
    ppo_results_path = "experiments/results/ppo/results.json"
    if os.path.exists(ppo_results_path):
        with open(ppo_results_path, 'r') as f:
            ppo_data = json.load(f)
        
        create_all_visualizations(
            ppo_data['training_stats'],
            ppo_data['evaluation'],
            ppo_data.get('convergence', {}),
            "experiments/results/ppo/plots",
            "PPO"
        )
        print("✓ PPO visualizations complete")
    else:
        print("✗ PPO results not found")
    
    # Comparison (Plot 3)
    print("\n" + "-" * 60)
    print("Regenerating comparison plot...")
    print("-" * 60)
    
    comparison_results_path = "experiments/results/comparison/comparison_results.json"
    if os.path.exists(comparison_results_path):
        with open(comparison_results_path, 'r') as f:
            comp_data = json.load(f)
        
        comparison_data = {
            'Q-Learning': (
                np.array(comp_data['q_learning']['training_stats']['episodes']),
                np.array(comp_data['q_learning']['training_stats']['rewards'])
            ),
            'PPO': (
                np.array(comp_data['ppo']['training_stats']['episodes']),
                np.array(comp_data['ppo']['training_stats']['rewards'])
            ),
        }
        
        plot_comparison(
            comparison_data,
            title="Q-Learning vs PPO: Learning Curves Comparison",
            save_path="experiments/results/comparison/3_algorithm_comparison.png",
            show=False
        )
        print("✓ Comparison plot complete")
    else:
        print("✗ Comparison results not found")
    
    print("\n" + "=" * 60)
    print("Visualization Regeneration Complete!")
    print("=" * 60)
    print("\nTotal visualization files created:")
    print("  - Q-Learning: 10 files")
    print("  - PPO: 10 files")
    print("  - Comparison: 1 file")
    print("  - Total: 21 files (11 unique visualizations)")
    print("\nAll files saved in experiments/results/*/plots/")


if __name__ == "__main__":
    regenerate_all()

