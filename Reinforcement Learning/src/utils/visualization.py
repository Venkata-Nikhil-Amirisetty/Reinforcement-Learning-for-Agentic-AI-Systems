"""
Visualization utilities for reinforcement learning results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Color palette
COLORS = {
    'q_learning': '#2E86AB',  # Blue
    'ppo': '#A23B72',        # Purple
    'dqn': '#F18F01',         # Orange
    'background': '#F5F5F5',
    'grid': '#E0E0E0'
}


def plot_learning_curve(
    episodes: np.ndarray,
    rewards: np.ndarray,
    title: str = "Learning Curve",
    save_path: Optional[str] = None,
    show: bool = True,
    color: str = '#2E86AB',
    show_std: bool = True
):
    """
    Plot learning curve showing reward over episodes with enhanced visualization.
    
    Args:
        episodes: Episode numbers
        rewards: Episode rewards
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
        color: Color for the plot
        show_std: Whether to show standard deviation band
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    # Plot raw rewards with transparency
    ax.plot(episodes, rewards, alpha=0.15, color=color, linewidth=0.5, label='Raw Rewards')
    
    # Plot moving average with confidence interval
    window_size = min(100, len(rewards) // 10)
    if window_size > 1:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_episodes = episodes[window_size-1:]
        
        # Calculate rolling std for confidence interval
        if show_std and len(rewards) > window_size:
            rolling_std = []
            for i in range(len(moving_avg)):
                start_idx = max(0, i)
                end_idx = min(len(rewards), i + window_size)
                rolling_std.append(np.std(rewards[start_idx:end_idx]))
            rolling_std = np.array(rolling_std)
            
            # Plot confidence interval
            ax.fill_between(
                moving_episodes,
                moving_avg - rolling_std,
                moving_avg + rolling_std,
                alpha=0.2,
                color=color,
                label='±1 Std Dev'
            )
        
        # Plot moving average
        ax.plot(
            moving_episodes,
            moving_avg,
            color=color,
            linewidth=2.5,
            label=f'Moving Average (window={window_size})',
            zorder=5
        )
    
    # Add final value annotation
    if len(rewards) > 0:
        final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        ax.axhline(y=final_avg, color=color, linestyle='--', alpha=0.5, linewidth=1.5, label=f'Final Avg: {final_avg:.3f}')
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Reward', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    # Improve layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Algorithm Comparison",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Compare multiple algorithms with enhanced visualization.
    
    Args:
        results: Dictionary mapping algorithm names to (episodes, rewards) tuples
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    # Color mapping
    color_map = {
        'Q-Learning': COLORS['q_learning'],
        'PPO': COLORS['ppo'],
        'DQN': COLORS['dqn']
    }
    
    # Statistics for annotation
    stats_text = []
    
    for name, (episodes, rewards) in results.items():
        color = color_map.get(name, COLORS['q_learning'])
        
        # Plot raw data with transparency
        ax.plot(episodes, rewards, alpha=0.1, color=color, linewidth=0.5)
        
        # Plot moving average
        window_size = min(100, len(rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            moving_episodes = episodes[window_size-1:]
            
            # Calculate confidence interval
            rolling_std = []
            for i in range(len(moving_avg)):
                start_idx = max(0, i)
                end_idx = min(len(rewards), i + window_size)
                rolling_std.append(np.std(rewards[start_idx:end_idx]))
            rolling_std = np.array(rolling_std)
            
            # Plot confidence band
            ax.fill_between(
                moving_episodes,
                moving_avg - rolling_std,
                moving_avg + rolling_std,
                alpha=0.15,
                color=color
            )
            
            # Plot moving average line
            ax.plot(
                moving_episodes,
                moving_avg,
                label=name,
                linewidth=3,
                color=color,
                zorder=5
            )
        else:
            ax.plot(episodes, rewards, label=name, linewidth=2.5, color=color, alpha=0.8)
        
        # Calculate and store statistics
        final_mean = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        final_std = np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards)
        stats_text.append(f"{name}: {final_mean:.3f} ± {final_std:.3f}")
    
    # Add statistics text box
    stats_str = '\n'.join(stats_text)
    ax.text(
        0.02, 0.98,
        stats_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace'
    )
    
    ax.set_xlabel('Episode', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Reward', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    # Improve layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_action_distribution(
    actions: List[int],
    action_names: List[str],
    title: str = "Action Distribution",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot distribution of actions taken.
    
    Args:
        actions: List of action indices
        action_names: Names for each action
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
    """
    plt.figure(figsize=(12, 6))
    
    action_counts = np.bincount(actions, minlength=len(action_names))
    action_probs = action_counts / len(actions) if len(actions) > 0 else action_counts
    
    plt.bar(range(len(action_names)), action_probs)
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title(title)
    plt.xticks(range(len(action_names)), action_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_evaluation_metrics(
    metrics: Dict[str, Any],
    title: str = "Evaluation Metrics",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot evaluation metrics with enhanced styling.
    
    Args:
        metrics: Dictionary of metrics
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.3)
    
    # Color scheme
    difficulty_colors = ['#4CAF50', '#FF9800', '#F44336']  # Green, Orange, Red
    scaffolding_colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    # Difficulty distribution
    if 'difficulty_distribution' in metrics:
        ax1 = fig.add_subplot(gs[0])
        diff_data = metrics['difficulty_distribution']
        keys = list(diff_data.keys())
        values = list(diff_data.values())
        
        bars = ax1.bar(keys, values, color=difficulty_colors[:len(keys)], edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Difficulty Level', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_title('Difficulty Level Distribution', fontweight='bold', pad=15)
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_facecolor('white')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
    
    # Scaffolding distribution
    if 'scaffolding_distribution' in metrics:
        ax2 = fig.add_subplot(gs[1])
        scaff_data = metrics['scaffolding_distribution']
        keys = list(scaff_data.keys())
        values = list(scaff_data.values())
        
        bars = ax2.bar(keys, values, color=scaffolding_colors[:len(keys)], edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Scaffolding Strategy', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Scaffolding Strategy Distribution', fontweight='bold', pad=15)
        ax2.tick_params(axis='x', labelrotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor('white')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', rotation=90)
    
    # Summary statistics
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    stats_text = "Evaluation Summary\n" + "="*20 + "\n\n"
    if 'mean_score' in metrics:
        stats_text += f"Mean Score:\n{metrics['mean_score']:.3f}\n\n"
    if 'std_score' in metrics:
        stats_text += f"Std Score:\n{metrics['std_score']:.3f}\n\n"
    if 'mean_time' in metrics:
        stats_text += f"Mean Time:\n{metrics['mean_time']:.1f}s\n"
    
    ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            family='monospace', fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_summary_plots(
    training_stats: Dict[str, List],
    evaluation_metrics: Dict[str, Any],
    output_dir: str = "experiments/results/plots",
    algorithm_name: str = None
):
    """
    Create all summary plots with enhanced styling.
    
    Args:
        training_stats: Training statistics
        evaluation_metrics: Evaluation metrics
        output_dir: Output directory for plots
        algorithm_name: Name of the algorithm (for color selection)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select color based on algorithm
    if algorithm_name:
        color = COLORS.get(algorithm_name.lower().replace('-', '_'), COLORS['q_learning'])
    else:
        color = COLORS['q_learning']
    
    # Learning curve
    if 'episodes' in training_stats and 'rewards' in training_stats:
        plot_learning_curve(
            np.array(training_stats['episodes']),
            np.array(training_stats['rewards']),
            title="Training Learning Curve",
            save_path=os.path.join(output_dir, "learning_curve.png"),
            show=False,
            color=color
        )
    
    # Student scores
    if 'episodes' in training_stats and 'student_scores' in training_stats:
        plot_learning_curve(
            np.array(training_stats['episodes']),
            np.array(training_stats['student_scores']),
            title="Student Performance Over Time",
            save_path=os.path.join(output_dir, "student_scores.png"),
            show=False,
            color=color
        )
    
    # Evaluation metrics
    plot_evaluation_metrics(
        evaluation_metrics,
        title="Evaluation Metrics",
        save_path=os.path.join(output_dir, "evaluation_metrics.png"),
        show=False
    )
    
    print(f"Plots saved to {output_dir}")

