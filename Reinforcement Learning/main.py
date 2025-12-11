"""
Main entry point for running experiments.
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning for Agentic AI Systems"
    )
    parser.add_argument(
        'experiment',
        choices=['q_learning', 'ppo', 'comparison', 'all'],
        help='Experiment to run'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes (default: 1000)'
    )
    parser.add_argument(
        '--students',
        type=int,
        default=10,
        help='Number of students per episode (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: experiments/results/{experiment})'
    )
    
    args = parser.parse_args()
    
    if args.experiment == 'q_learning':
        from experiments.run_q_learning_experiment import run_experiment
        output_dir = args.output or "experiments/results/q_learning"
        run_experiment(
            num_episodes=args.episodes,
            students_per_episode=args.students,
            output_dir=output_dir
        )
    
    elif args.experiment == 'ppo':
        from experiments.run_ppo_experiment import run_experiment
        output_dir = args.output or "experiments/results/ppo"
        run_experiment(
            num_episodes=args.episodes,
            students_per_episode=args.students,
            output_dir=output_dir
        )
    
    elif args.experiment == 'comparison':
        from experiments.run_comparison import run_comparison
        output_dir = args.output or "experiments/results/comparison"
        run_comparison(
            num_episodes=args.episodes,
            students_per_episode=args.students,
            output_dir=output_dir
        )
    
    elif args.experiment == 'all':
        print("Running all experiments...")
        from experiments.run_q_learning_experiment import run_experiment as run_ql
        from experiments.run_ppo_experiment import run_experiment as run_ppo
        from experiments.run_comparison import run_comparison
        
        print("\n" + "="*60)
        print("1. Q-Learning Experiment")
        print("="*60)
        run_ql(
            num_episodes=args.episodes,
            students_per_episode=args.students,
            output_dir="experiments/results/q_learning"
        )
        
        print("\n" + "="*60)
        print("2. PPO Experiment")
        print("="*60)
        run_ppo(
            num_episodes=args.episodes,
            students_per_episode=args.students,
            output_dir="experiments/results/ppo"
        )
        
        print("\n" + "="*60)
        print("3. Comparison")
        print("="*60)
        run_comparison(
            num_episodes=args.episodes,
            students_per_episode=args.students,
            output_dir="experiments/results/comparison"
        )
        
        print("\n" + "="*60)
        print("All experiments completed!")
        print("="*60)


if __name__ == "__main__":
    main()

