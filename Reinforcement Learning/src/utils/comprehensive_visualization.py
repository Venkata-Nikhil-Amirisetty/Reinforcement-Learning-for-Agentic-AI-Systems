"""
Comprehensive visualization suite for reinforcement learning results.
Creates 11 visualization files matching professional standards.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
from .visualization import COLORS, plot_learning_curve, plot_comparison

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Moving average window
MOVING_AVG_WINDOW = 50


def plot_1_reward_learning_curve(
    episodes: np.ndarray,
    rewards: np.ndarray,
    save_path: str,
    algorithm_name: str = "Algorithm",
    color: str = None
):
    """Plot 1: Reward Learning Curve with raw data and moving average."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    if color is None:
        color = COLORS.get(algorithm_name.lower().replace('-', '_'), COLORS['q_learning'])
    
    # Raw data (semi-transparent)
    ax.plot(episodes, rewards, alpha=0.2, color=color, linewidth=0.8, label='Raw Rewards')
    
    # Moving average (50-episode window)
    if len(rewards) >= MOVING_AVG_WINDOW:
        moving_avg = np.convolve(rewards, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid')
        moving_episodes = episodes[MOVING_AVG_WINDOW-1:]
        ax.plot(moving_episodes, moving_avg, color=color, linewidth=3, 
                label=f'Moving Average ({MOVING_AVG_WINDOW}-episode window)', zorder=5)
    
    ax.set_xlabel('Episode', fontweight='bold', fontsize=13)
    ax.set_ylabel('Reward', fontweight='bold', fontsize=13)
    ax.set_title(f'{algorithm_name}: Reward Learning Curve', fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_2_student_score_learning_curve(
    episodes: np.ndarray,
    scores: np.ndarray,
    save_path: str,
    algorithm_name: str = "Algorithm",
    color: str = None
):
    """Plot 2: Student Score Learning Curve."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    if color is None:
        color = COLORS.get(algorithm_name.lower().replace('-', '_'), COLORS['q_learning'])
    
    # Raw data
    ax.plot(episodes, scores, alpha=0.2, color=color, linewidth=0.8, label='Raw Scores')
    
    # Moving average
    if len(scores) >= MOVING_AVG_WINDOW:
        moving_avg = np.convolve(scores, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid')
        moving_episodes = episodes[MOVING_AVG_WINDOW-1:]
        ax.plot(moving_episodes, moving_avg, color=color, linewidth=3,
                label=f'Moving Average ({MOVING_AVG_WINDOW}-episode window)', zorder=5)
    
    ax.set_xlabel('Episode', fontweight='bold', fontsize=13)
    ax.set_ylabel('Student Score', fontweight='bold', fontsize=13)
    ax.set_title(f'{algorithm_name}: Student Performance Learning Curve', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_3_algorithm_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str
):
    """Plot 3: Algorithm Comparison."""
    plot_comparison(results, "Algorithm Comparison: Learning Curves", save_path, show=False)


def plot_4_convergence_analysis(
    episodes: np.ndarray,
    rewards: np.ndarray,
    save_path: str,
    algorithm_name: str = "Algorithm",
    color: str = None
):
    """Plot 4: Convergence Behavior Analysis."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    if color is None:
        color = COLORS.get(algorithm_name.lower().replace('-', '_'), COLORS['q_learning'])
    
    # Top: Learning curve with convergence zones
    ax1 = axes[0]
    ax1.plot(episodes, rewards, alpha=0.2, color=color, linewidth=0.8, label='Raw Rewards')
    
    if len(rewards) >= MOVING_AVG_WINDOW:
        moving_avg = np.convolve(rewards, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid')
        moving_episodes = episodes[MOVING_AVG_WINDOW-1:]
        ax1.plot(moving_episodes, moving_avg, color=color, linewidth=3,
                label=f'Moving Average ({MOVING_AVG_WINDOW}-episode window)', zorder=5)
        
        # Mark convergence zones
        if len(moving_avg) > 200:
            early_avg = np.mean(moving_avg[:100])
            late_avg = np.mean(moving_avg[-100:])
            ax1.axhline(y=early_avg, color='red', linestyle='--', alpha=0.5, 
                       label=f'Early Avg: {early_avg:.3f}')
            ax1.axhline(y=late_avg, color='green', linestyle='--', alpha=0.5,
                       label=f'Late Avg: {late_avg:.3f}')
    
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Reward', fontweight='bold')
    ax1.set_title(f'{algorithm_name}: Convergence Analysis', fontweight='bold', fontsize=14)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('white')
    
    # Bottom: Variance over time
    ax2 = axes[1]
    if len(rewards) >= 100:
        window = 100
        variances = []
        variance_episodes = []
        for i in range(0, len(rewards) - window + 1, window // 4):
            window_rewards = rewards[i:i+window]
            variances.append(np.var(window_rewards))
            variance_episodes.append(episodes[i + window // 2])
        
        ax2.plot(variance_episodes, variances, color=color, linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Reward Variance', fontweight='bold')
        ax2.set_title('Reward Variance Over Time (Convergence Indicator)', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_5_performance_distribution(
    rewards: np.ndarray,
    scores: np.ndarray,
    save_path: str,
    algorithm_name: str = "Algorithm",
    color: str = None
):
    """Plot 5: Performance Distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    if color is None:
        color = COLORS.get(algorithm_name.lower().replace('-', '_'), COLORS['q_learning'])
    
    # Reward distribution
    ax1 = axes[0]
    ax1.hist(rewards, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(rewards):.3f}')
    ax1.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(rewards):.3f}')
    ax1.set_xlabel('Reward', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Reward Distribution', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('white')
    
    # Score distribution
    ax2 = axes[1]
    ax2.hist(scores, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(scores):.3f}')
    ax2.axvline(np.median(scores), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(scores):.3f}')
    ax2.set_xlabel('Student Score', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Student Score Distribution', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('white')
    ax2.set_xlim([0, 1])
    
    fig.suptitle(f'{algorithm_name}: Performance Distributions', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_6_difficulty_scaffolding_patterns(
    evaluation_metrics: Dict[str, Any],
    save_path: str,
    algorithm_name: str = "Algorithm"
):
    """Plot 6: Topic-Specific Patterns (Difficulty & Scaffolding)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    # Difficulty distribution
    if 'difficulty_distribution' in evaluation_metrics:
        ax1 = axes[0]
        diff_data = evaluation_metrics['difficulty_distribution']
        keys = list(diff_data.keys())
        values = list(diff_data.values())
        colors = ['#4CAF50', '#FF9800', '#F44336'][:len(keys)]
        
        bars = ax1.bar(keys, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Difficulty Level', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_title('Difficulty Level Usage Pattern', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_facecolor('white')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Scaffolding distribution
    if 'scaffolding_distribution' in evaluation_metrics:
        ax2 = axes[1]
        scaff_data = evaluation_metrics['scaffolding_distribution']
        keys = list(scaff_data.keys())
        values = list(scaff_data.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(keys)))
        
        bars = ax2.bar(keys, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Scaffolding Strategy', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Scaffolding Strategy Usage Pattern', fontweight='bold', fontsize=14)
        ax2.tick_params(axis='x', labelrotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor('white')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', 
                    fontweight='bold', rotation=90)
    
    fig.suptitle(f'{algorithm_name}: Teaching Strategy Patterns', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_7_reward_vs_score_correlation(
    rewards: np.ndarray,
    scores: np.ndarray,
    save_path: str,
    algorithm_name: str = "Algorithm",
    color: str = None
):
    """Plot 7: Reward vs Score Correlation."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    if color is None:
        color = COLORS.get(algorithm_name.lower().replace('-', '_'), COLORS['q_learning'])
    
    # Scatter plot
    ax.scatter(rewards, scores, alpha=0.5, color=color, s=20, edgecolors='black', linewidth=0.5)
    
    # Correlation line
    correlation = np.corrcoef(rewards, scores)[0, 1]
    z = np.polyfit(rewards, scores, 1)
    p = np.poly1d(z)
    ax.plot(rewards, p(rewards), "r--", alpha=0.8, linewidth=2,
           label=f'Correlation: {correlation:.3f}')
    
    ax.set_xlabel('Reward', fontweight='bold', fontsize=13)
    ax.set_ylabel('Student Score', fontweight='bold', fontsize=13)
    ax.set_title(f'{algorithm_name}: Reward vs Student Score Correlation', 
                 fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_8_training_vs_evaluation(
    training_stats: Dict[str, List],
    evaluation_metrics: Dict[str, Any],
    save_path: str,
    algorithm_name: str = "Algorithm",
    color: str = None
):
    """Plot 8: Training vs Evaluation Comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    if color is None:
        color = COLORS.get(algorithm_name.lower().replace('-', '_'), COLORS['q_learning'])
    
    # Training performance
    if 'rewards' in training_stats and 'student_scores' in training_stats:
        ax1 = axes[0]
        episodes = np.array(training_stats['episodes'])
        rewards = np.array(training_stats['rewards'])
        scores = np.array(training_stats['student_scores'])
        
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(episodes, rewards, color=color, linewidth=2, label='Training Reward', alpha=0.7)
        line2 = ax1_twin.plot(episodes, scores, color='orange', linewidth=2, label='Training Score', alpha=0.7)
        
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Reward', fontweight='bold', color=color)
        ax1_twin.set_ylabel('Student Score', fontweight='bold', color='orange')
        ax1.set_title('Training Performance', fontweight='bold', fontsize=14)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1_twin.tick_params(axis='y', labelcolor='orange')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('white')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', framealpha=0.9)
    
    # Evaluation summary
    ax2 = axes[1]
    if 'mean_score' in evaluation_metrics:
        metrics_to_plot = {
            'Mean Score': evaluation_metrics.get('mean_score', 0),
            'Std Score': evaluation_metrics.get('std_score', 0),
        }
        
        bars = ax2.bar(metrics_to_plot.keys(), metrics_to_plot.values(), 
                      color=[color, 'orange'], edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_title('Evaluation Metrics', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor('white')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle(f'{algorithm_name}: Training vs Evaluation', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_9_statistical_summary(
    training_stats: Dict[str, List],
    evaluation_metrics: Dict[str, Any],
    convergence: Dict[str, Any],
    save_path: str,
    algorithm_name: str = "Algorithm"
):
    """Plot 9: Statistical Summary Dashboard."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Summary statistics text
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    stats_text = f"{algorithm_name} - Statistical Summary\n" + "="*60 + "\n\n"
    
    if 'rewards' in training_stats:
        rewards = np.array(training_stats['rewards'])
        stats_text += f"Training Rewards:\n"
        stats_text += f"  Mean: {np.mean(rewards):.4f}\n"
        stats_text += f"  Std:  {np.std(rewards):.4f}\n"
        stats_text += f"  Min:  {np.min(rewards):.4f}\n"
        stats_text += f"  Max:  {np.max(rewards):.4f}\n\n"
    
    if 'student_scores' in training_stats:
        scores = np.array(training_stats['student_scores'])
        stats_text += f"Student Scores:\n"
        stats_text += f"  Mean: {np.mean(scores):.4f}\n"
        stats_text += f"  Std:  {np.std(scores):.4f}\n"
        stats_text += f"  Min:  {np.min(scores):.4f}\n"
        stats_text += f"  Max:  {np.max(scores):.4f}\n\n"
    
    if 'mean_score' in evaluation_metrics:
        stats_text += f"Evaluation:\n"
        stats_text += f"  Mean Score: {evaluation_metrics['mean_score']:.4f}\n"
        stats_text += f"  Std Score:  {evaluation_metrics['std_score']:.4f}\n\n"
    
    if convergence:
        stats_text += f"Convergence:\n"
        stats_text += f"  Converged: {convergence.get('converged', False)}\n"
        if 'improvement' in convergence:
            stats_text += f"  Improvement: {convergence['improvement']:.4f}\n"
        if 'improvement_ratio' in convergence:
            stats_text += f"  Improvement Ratio: {convergence['improvement_ratio']:.4f}\n"
    
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Box plots
    ax2 = fig.add_subplot(gs[1, 0])
    if 'rewards' in training_stats and 'student_scores' in training_stats:
        data = [training_stats['rewards'], training_stats['student_scores']]
        bp = ax2.boxplot(data, labels=['Rewards', 'Scores'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['q_learning'])
        bp['boxes'][1].set_facecolor(COLORS['ppo'])
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_title('Distribution Summary', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor('white')
    
    # Convergence metrics
    ax3 = fig.add_subplot(gs[1, 1])
    if convergence and 'early_avg' in convergence and 'late_avg' in convergence:
        metrics = ['Early Avg', 'Late Avg', 'Improvement']
        values = [
            convergence.get('early_avg', 0),
            convergence.get('late_avg', 0),
            convergence.get('improvement', 0)
        ]
        colors_bar = ['red', 'green', 'blue']
        bars = ax3.bar(metrics, values, color=colors_bar, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax3.set_ylabel('Value', fontweight='bold')
        ax3.set_title('Convergence Metrics', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_facecolor('white')
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'{algorithm_name}: Complete Statistical Summary', 
                 fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_10_episode_wise_analysis(
    episodes: np.ndarray,
    rewards: np.ndarray,
    scores: np.ndarray,
    save_path: str,
    algorithm_name: str = "Algorithm",
    color: str = None
):
    """Plot 10: Episode-wise Detailed Analysis."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.patch.set_facecolor('white')
    
    if color is None:
        color = COLORS.get(algorithm_name.lower().replace('-', '_'), COLORS['q_learning'])
    
    # Rewards over episodes
    ax1 = axes[0]
    ax1.plot(episodes, rewards, alpha=0.3, color=color, linewidth=0.8)
    if len(rewards) >= MOVING_AVG_WINDOW:
        moving_avg = np.convolve(rewards, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid')
        moving_episodes = episodes[MOVING_AVG_WINDOW-1:]
        ax1.plot(moving_episodes, moving_avg, color=color, linewidth=2.5, zorder=5)
    ax1.set_ylabel('Reward', fontweight='bold')
    ax1.set_title('Reward per Episode', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('white')
    
    # Scores over episodes
    ax2 = axes[1]
    ax2.plot(episodes, scores, alpha=0.3, color='orange', linewidth=0.8)
    if len(scores) >= MOVING_AVG_WINDOW:
        moving_avg = np.convolve(scores, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid')
        moving_episodes = episodes[MOVING_AVG_WINDOW-1:]
        ax2.plot(moving_episodes, moving_avg, color='orange', linewidth=2.5, zorder=5)
    ax2.set_ylabel('Student Score', fontweight='bold')
    ax2.set_title('Student Score per Episode', fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('white')
    ax2.set_ylim([0, 1])
    
    # Combined view
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(episodes, rewards, color=color, linewidth=1.5, alpha=0.7, label='Reward')
    line2 = ax3_twin.plot(episodes, scores, color='orange', linewidth=1.5, alpha=0.7, label='Score')
    ax3.set_xlabel('Episode', fontweight='bold')
    ax3.set_ylabel('Reward', fontweight='bold', color=color)
    ax3_twin.set_ylabel('Student Score', fontweight='bold', color='orange')
    ax3.set_title('Combined View: Reward and Score', fontweight='bold', fontsize=13)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('white')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='best', framealpha=0.9)
    
    fig.suptitle(f'{algorithm_name}: Episode-wise Analysis', 
                 fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_11_comprehensive_dashboard(
    training_stats: Dict[str, List],
    evaluation_metrics: Dict[str, Any],
    save_path: str,
    algorithm_name: str = "Algorithm",
    color: str = None
):
    """Plot 11: Comprehensive Dashboard."""
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    if color is None:
        color = COLORS.get(algorithm_name.lower().replace('-', '_'), COLORS['q_learning'])
    
    episodes = np.array(training_stats.get('episodes', []))
    rewards = np.array(training_stats.get('rewards', []))
    scores = np.array(training_stats.get('student_scores', []))
    
    # 1. Learning curve (top left, span 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(episodes, rewards, alpha=0.2, color=color, linewidth=0.8)
    if len(rewards) >= MOVING_AVG_WINDOW:
        moving_avg = np.convolve(rewards, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid')
        moving_episodes = episodes[MOVING_AVG_WINDOW-1:]
        ax1.plot(moving_episodes, moving_avg, color=color, linewidth=2.5, zorder=5)
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Reward', fontweight='bold')
    ax1.set_title('Learning Curve', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('white')
    
    # 2. Key metrics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    metrics_text = "Key Metrics\n" + "="*20 + "\n\n"
    if len(rewards) > 0:
        metrics_text += f"Final Reward:\n{np.mean(rewards[-100:]):.3f}\n\n"
    if len(scores) > 0:
        metrics_text += f"Final Score:\n{np.mean(scores[-100:]):.3f}\n\n"
    if 'mean_score' in evaluation_metrics:
        metrics_text += f"Eval Score:\n{evaluation_metrics['mean_score']:.3f}\n"
    ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontweight='bold')
    
    # 3. Reward distribution (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    if len(rewards) > 0:
        ax3.hist(rewards, bins=30, color=color, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Reward', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Reward Distribution', fontweight='bold', fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_facecolor('white')
    
    # 4. Score distribution (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    if len(scores) > 0:
        ax4.hist(scores, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Score', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Score Distribution', fontweight='bold', fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_facecolor('white')
        ax4.set_xlim([0, 1])
    
    # 5. Difficulty distribution (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    if 'difficulty_distribution' in evaluation_metrics:
        diff_data = evaluation_metrics['difficulty_distribution']
        ax5.bar(diff_data.keys(), diff_data.values(), 
               color=['#4CAF50', '#FF9800', '#F44336'][:len(diff_data)], 
               edgecolor='black', alpha=0.8)
        ax5.set_xlabel('Difficulty', fontweight='bold')
        ax5.set_ylabel('Count', fontweight='bold')
        ax5.set_title('Difficulty Usage', fontweight='bold', fontsize=11)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_facecolor('white')
    
    # 6. Correlation (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    if len(rewards) > 0 and len(scores) > 0:
        ax6.scatter(rewards, scores, alpha=0.5, color=color, s=10, edgecolors='black', linewidth=0.3)
        correlation = np.corrcoef(rewards, scores)[0, 1]
        ax6.set_xlabel('Reward', fontweight='bold')
        ax6.set_ylabel('Score', fontweight='bold')
        ax6.set_title(f'Correlation: {correlation:.3f}', fontweight='bold', fontsize=11)
        ax6.grid(True, alpha=0.3)
        ax6.set_facecolor('white')
    
    # 7. Scaffolding distribution (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])
    if 'scaffolding_distribution' in evaluation_metrics:
        scaff_data = evaluation_metrics['scaffolding_distribution']
        ax7.bar(range(len(scaff_data)), list(scaff_data.values()),
               color=plt.cm.Set3(np.linspace(0, 1, len(scaff_data))),
               edgecolor='black', alpha=0.8)
        ax7.set_xticks(range(len(scaff_data)))
        ax7.set_xticklabels(list(scaff_data.keys()), rotation=45, ha='right')
        ax7.set_ylabel('Count', fontweight='bold')
        ax7.set_title('Scaffolding Usage', fontweight='bold', fontsize=11)
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_facecolor('white')
    
    # 8. Performance over time (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    if len(rewards) >= MOVING_AVG_WINDOW and len(scores) >= MOVING_AVG_WINDOW:
        moving_rewards = np.convolve(rewards, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid')
        moving_scores = np.convolve(scores, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode='valid')
        moving_episodes = episodes[MOVING_AVG_WINDOW-1:]
        ax8_twin = ax8.twinx()
        line1 = ax8.plot(moving_episodes, moving_rewards, color=color, linewidth=2, label='Reward')
        line2 = ax8_twin.plot(moving_episodes, moving_scores, color='orange', linewidth=2, label='Score')
        ax8.set_xlabel('Episode', fontweight='bold')
        ax8.set_ylabel('Reward', fontweight='bold', color=color)
        ax8_twin.set_ylabel('Score', fontweight='bold', color='orange')
        ax8.set_title('Performance Trends', fontweight='bold', fontsize=11)
        ax8.tick_params(axis='y', labelcolor=color)
        ax8_twin.tick_params(axis='y', labelcolor='orange')
        ax8.grid(True, alpha=0.3)
        ax8.set_facecolor('white')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax8.legend(lines, labels, loc='best', fontsize=8)
    
    plt.suptitle(f'{algorithm_name}: Comprehensive Dashboard', 
                 fontweight='bold', fontsize=18, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_all_visualizations(
    training_stats: Dict[str, List],
    evaluation_metrics: Dict[str, Any],
    convergence: Dict[str, Any],
    output_dir: str,
    algorithm_name: str = "Algorithm"
):
    """
    Create all 11 comprehensive visualization files.
    
    Args:
        training_stats: Training statistics dictionary
        evaluation_metrics: Evaluation metrics dictionary
        convergence: Convergence metrics dictionary
        output_dir: Output directory for plots
        algorithm_name: Name of the algorithm
    """
    os.makedirs(output_dir, exist_ok=True)
    
    episodes = np.array(training_stats.get('episodes', []))
    rewards = np.array(training_stats.get('rewards', []))
    scores = np.array(training_stats.get('student_scores', []))
    
    color = COLORS.get(algorithm_name.lower().replace('-', '_').replace(' ', '_'), COLORS['q_learning'])
    
    print(f"Creating comprehensive visualizations for {algorithm_name}...")
    
    # Plot 1: Reward Learning Curve
    plot_1_reward_learning_curve(
        episodes, rewards,
        os.path.join(output_dir, "1_reward_learning_curve.png"),
        algorithm_name, color
    )
    
    # Plot 2: Student Score Learning Curve
    plot_2_student_score_learning_curve(
        episodes, scores,
        os.path.join(output_dir, "2_student_score_learning_curve.png"),
        algorithm_name, color
    )
    
    # Plot 4: Convergence Analysis
    plot_4_convergence_analysis(
        episodes, rewards,
        os.path.join(output_dir, "4_convergence_analysis.png"),
        algorithm_name, color
    )
    
    # Plot 5: Performance Distribution
    plot_5_performance_distribution(
        rewards, scores,
        os.path.join(output_dir, "5_performance_distribution.png"),
        algorithm_name, color
    )
    
    # Plot 6: Difficulty & Scaffolding Patterns
    plot_6_difficulty_scaffolding_patterns(
        evaluation_metrics,
        os.path.join(output_dir, "6_teaching_strategy_patterns.png"),
        algorithm_name
    )
    
    # Plot 7: Reward vs Score Correlation
    plot_7_reward_vs_score_correlation(
        rewards, scores,
        os.path.join(output_dir, "7_reward_score_correlation.png"),
        algorithm_name, color
    )
    
    # Plot 8: Training vs Evaluation
    plot_8_training_vs_evaluation(
        training_stats, evaluation_metrics,
        os.path.join(output_dir, "8_training_vs_evaluation.png"),
        algorithm_name, color
    )
    
    # Plot 9: Statistical Summary
    plot_9_statistical_summary(
        training_stats, evaluation_metrics, convergence,
        os.path.join(output_dir, "9_statistical_summary.png"),
        algorithm_name
    )
    
    # Plot 10: Episode-wise Analysis
    plot_10_episode_wise_analysis(
        episodes, rewards, scores,
        os.path.join(output_dir, "10_episode_wise_analysis.png"),
        algorithm_name, color
    )
    
    # Plot 11: Comprehensive Dashboard
    plot_11_comprehensive_dashboard(
        training_stats, evaluation_metrics,
        os.path.join(output_dir, "11_comprehensive_dashboard.png"),
        algorithm_name, color
    )
    
    print(f"✓ Created 10 visualization files in {output_dir}")
    print(f"  (Plot 3 is comparison plot, created separately)")

