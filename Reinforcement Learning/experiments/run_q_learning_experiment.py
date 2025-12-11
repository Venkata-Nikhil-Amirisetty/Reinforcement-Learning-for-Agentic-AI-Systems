"""
Experiment script for Q-Learning tutorial agent.
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.tutorial_agent import TutorialAgent
from src.utils.metrics import MetricsTracker, compute_convergence_metrics
from src.utils.visualization import plot_learning_curve, create_summary_plots
from src.utils.comprehensive_visualization import create_all_visualizations
import json


def run_experiment(
    num_episodes: int = 1000,
    students_per_episode: int = 10,
    output_dir: str = "experiments/results/q_learning"
):
    """
    Run Q-Learning experiment.
    
    Args:
        num_episodes: Number of training episodes
        students_per_episode: Number of students per episode
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Q-Learning Tutorial Agent Experiment")
    print("=" * 60)
    
    # Initialize agent
    print("\nInitializing Q-Learning agent...")
    agent = TutorialAgent(
        rl_algorithm='q_learning',
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )
    
    # Initialize metrics tracker
    metrics = MetricsTracker(window_size=100)
    
    # Train agent
    print(f"\nTraining for {num_episodes} episodes...")
    agent.train(num_episodes=num_episodes, students_per_episode=students_per_episode)
    
    # Extract training statistics
    episodes = np.array(agent.training_stats['episodes'])
    rewards = np.array(agent.training_stats['rewards'])
    scores = np.array(agent.training_stats['student_scores'])
    
    # Compute convergence metrics
    convergence = compute_convergence_metrics(rewards.tolist())
    
    # Evaluate agent
    print("\nEvaluating agent...")
    evaluation_results = agent.evaluate(num_students=50)
    
    # Save results
    results = {
        'algorithm': 'q_learning',
        'num_episodes': num_episodes,
        'students_per_episode': students_per_episode,
        'training_stats': {
            'episodes': episodes.tolist(),
            'rewards': rewards.tolist(),
            'student_scores': scores.tolist(),
        },
        'convergence': convergence,
        'evaluation': evaluation_results,
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Create comprehensive visualizations (11 files)
    print("\nCreating comprehensive visualizations...")
    create_all_visualizations(
        agent.training_stats,
        evaluation_results,
        convergence,
        output_dir=os.path.join(output_dir, 'plots'),
        algorithm_name='Q-Learning'
    )
    
    # Also create summary plots for backward compatibility
    create_summary_plots(
        agent.training_stats,
        evaluation_results,
        output_dir=os.path.join(output_dir, 'plots'),
        algorithm_name='q_learning'
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"Final Average Reward: {rewards[-100:].mean():.3f} ± {rewards[-100:].std():.3f}")
    print(f"Final Average Score: {scores[-100:].mean():.3f} ± {scores[-100:].std():.3f}")
    print(f"Evaluation Mean Score: {evaluation_results['mean_score']:.3f} ± {evaluation_results['std_score']:.3f}")
    print(f"Converged: {convergence['converged']}")
    if convergence['converged']:
        print(f"Improvement: {convergence['improvement']:.3f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Q-Learning experiment")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--students", type=int, default=10, help="Students per episode")
    parser.add_argument("--output", type=str, default="experiments/results/q_learning", help="Output directory")
    
    args = parser.parse_args()
    
    run_experiment(
        num_episodes=args.episodes,
        students_per_episode=args.students,
        output_dir=args.output
    )

