"""
Compare Q-Learning and PPO algorithms.
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.tutorial_agent import TutorialAgent
from src.utils.metrics import compute_convergence_metrics, compare_agents
from src.utils.visualization import plot_comparison
import json


def run_comparison(
    num_episodes: int = 1000,
    students_per_episode: int = 10,
    output_dir: str = "experiments/results/comparison"
):
    """
    Compare Q-Learning and PPO algorithms.
    
    Args:
        num_episodes: Number of training episodes
        students_per_episode: Number of students per episode
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Algorithm Comparison: Q-Learning vs PPO")
    print("=" * 60)
    
    results = {}
    
    # Run Q-Learning
    print("\n" + "-" * 60)
    print("Training Q-Learning Agent...")
    print("-" * 60)
    ql_agent = TutorialAgent(
        rl_algorithm='q_learning',
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )
    ql_agent.train(num_episodes=num_episodes, students_per_episode=students_per_episode)
    ql_eval = ql_agent.evaluate(num_students=50)
    ql_convergence = compute_convergence_metrics(ql_agent.training_stats['rewards'])
    
    results['q_learning'] = {
        'training_stats': ql_agent.training_stats,
        'evaluation': ql_eval,
        'convergence': ql_convergence,
        'mean_reward': np.mean(ql_agent.training_stats['rewards'][-100:]),
        'std_reward': np.std(ql_agent.training_stats['rewards'][-100:]),
        'mean_score': ql_eval['mean_score'],
    }
    
    # Run PPO
    print("\n" + "-" * 60)
    print("Training PPO Agent...")
    print("-" * 60)
    ppo_agent = TutorialAgent(
        rl_algorithm='ppo',
        learning_rate=3e-4,
        discount_factor=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        update_epochs=10,
        batch_size=64,
    )
    ppo_agent.train(num_episodes=num_episodes, students_per_episode=students_per_episode)
    ppo_eval = ppo_agent.evaluate(num_students=50)
    ppo_convergence = compute_convergence_metrics(ppo_agent.training_stats['rewards'])
    
    results['ppo'] = {
        'training_stats': ppo_agent.training_stats,
        'evaluation': ppo_eval,
        'convergence': ppo_convergence,
        'mean_reward': np.mean(ppo_agent.training_stats['rewards'][-100:]),
        'std_reward': np.std(ppo_agent.training_stats['rewards'][-100:]),
        'mean_score': ppo_eval['mean_score'],
    }
    
    # Save results
    results_path = os.path.join(output_dir, 'comparison_results.json')
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for alg_name, alg_results in results.items():
        serializable_results[alg_name] = {
            'training_stats': {
                'episodes': alg_results['training_stats']['episodes'],
                'rewards': alg_results['training_stats']['rewards'],
                'student_scores': alg_results['training_stats']['student_scores'],
            },
            'evaluation': alg_results['evaluation'],
            'convergence': alg_results['convergence'],
            'mean_reward': float(alg_results['mean_reward']),
            'std_reward': float(alg_results['std_reward']),
            'mean_score': float(alg_results['mean_score']),
        }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Create comparison plot (Plot 3)
    print("\nCreating comparison plot (Plot 3)...")
    comparison_data = {
        'Q-Learning': (
            np.array(results['q_learning']['training_stats']['episodes']),
            np.array(results['q_learning']['training_stats']['rewards'])
        ),
        'PPO': (
            np.array(results['ppo']['training_stats']['episodes']),
            np.array(results['ppo']['training_stats']['rewards'])
        ),
    }
    
    plot_comparison(
        comparison_data,
        title="Q-Learning vs PPO: Learning Curves Comparison",
        save_path=os.path.join(output_dir, '3_algorithm_comparison.png'),
        show=False
    )
    
    # Also save as comparison_plot.png for backward compatibility
    plot_comparison(
        comparison_data,
        title="Q-Learning vs PPO: Learning Curves",
        save_path=os.path.join(output_dir, 'comparison_plot.png'),
        show=False
    )
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    
    print("\nQ-Learning:")
    print(f"  Final Reward: {results['q_learning']['mean_reward']:.3f} ± {results['q_learning']['std_reward']:.3f}")
    print(f"  Evaluation Score: {results['q_learning']['mean_score']:.3f}")
    print(f"  Converged: {results['q_learning']['convergence']['converged']}")
    
    print("\nPPO:")
    print(f"  Final Reward: {results['ppo']['mean_reward']:.3f} ± {results['ppo']['std_reward']:.3f}")
    print(f"  Evaluation Score: {results['ppo']['mean_score']:.3f}")
    print(f"  Converged: {results['ppo']['convergence']['converged']}")
    
    # Determine winner
    if results['q_learning']['mean_reward'] > results['ppo']['mean_reward']:
        winner = "Q-Learning"
        margin = results['q_learning']['mean_reward'] - results['ppo']['mean_reward']
    else:
        winner = "PPO"
        margin = results['ppo']['mean_reward'] - results['q_learning']['mean_reward']
    
    print(f"\nWinner: {winner} (margin: {margin:.3f})")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Q-Learning and PPO")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--students", type=int, default=10, help="Students per episode")
    parser.add_argument("--output", type=str, default="experiments/results/comparison", help="Output directory")
    
    args = parser.parse_args()
    
    run_comparison(
        num_episodes=args.episodes,
        students_per_episode=args.students,
        output_dir=args.output
    )

