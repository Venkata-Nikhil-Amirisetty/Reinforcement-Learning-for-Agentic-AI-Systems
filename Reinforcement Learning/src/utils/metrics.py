"""
Metrics and evaluation utilities for reinforcement learning agents.
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque


class MetricsTracker:
    """Track and compute metrics for RL training."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_lengths = deque(maxlen=self.window_size)
        self.recent_scores = deque(maxlen=self.window_size)
    
    def record_episode(
        self,
        reward: float,
        length: int,
        score: float = None
    ):
        """
        Record episode metrics.
        
        Args:
            reward: Total episode reward
            length: Episode length
            score: Additional score metric (e.g., student performance)
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.recent_rewards.append(reward)
        self.recent_lengths.append(length)
        
        if score is not None:
            self.episode_scores.append(score)
            self.recent_scores.append(score)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get current statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        if len(self.episode_rewards) > 0:
            stats['mean_reward'] = np.mean(self.recent_rewards)
            stats['std_reward'] = np.std(self.recent_rewards)
            stats['min_reward'] = np.min(self.recent_rewards)
            stats['max_reward'] = np.max(self.recent_rewards)
            
            stats['mean_length'] = np.mean(self.recent_lengths)
            stats['std_length'] = np.std(self.recent_lengths)
        
        if len(self.episode_scores) > 0:
            stats['mean_score'] = np.mean(self.recent_scores)
            stats['std_score'] = np.std(self.recent_scores)
        
        return stats
    
    def get_learning_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get learning curve data.
        
        Returns:
            (episodes, rewards) arrays
        """
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        rewards = np.array(self.episode_rewards)
        return episodes, rewards


def compute_convergence_metrics(
    rewards: List[float],
    window_size: int = 100
) -> Dict[str, Any]:
    """
    Compute convergence metrics for learning.
    
    Args:
        rewards: List of episode rewards
        window_size: Window size for stability check
        
    Returns:
        Convergence metrics
    """
    if len(rewards) < window_size * 2:
        return {'converged': False, 'reason': 'insufficient_data'}
    
    rewards = np.array(rewards)
    
    # Compute moving averages
    early_avg = np.mean(rewards[:window_size])
    late_avg = np.mean(rewards[-window_size:])
    
    # Compute variance
    early_var = np.var(rewards[:window_size])
    late_var = np.var(rewards[-window_size:])
    
    # Check convergence
    improvement = late_avg - early_avg
    improvement_ratio = improvement / (abs(early_avg) + 1e-8)
    variance_ratio = late_var / (early_var + 1e-8)
    
    # Convergence criteria
    converged = bool(
        improvement_ratio > 0.1 and  # Significant improvement
        variance_ratio < 2.0  # Stable variance
    )
    
    return {
        'converged': converged,
        'improvement': float(improvement),
        'improvement_ratio': float(improvement_ratio),
        'early_avg': float(early_avg),
        'late_avg': float(late_avg),
        'early_var': float(early_var),
        'late_var': float(late_var),
        'variance_ratio': float(variance_ratio),
    }


def compute_policy_entropy(actions: List[int], num_actions: int) -> float:
    """
    Compute entropy of action distribution (exploration measure).
    
    Args:
        actions: List of action indices
        num_actions: Number of possible actions
        
    Returns:
        Entropy value
    """
    if len(actions) == 0:
        return 0.0
    
    action_counts = np.bincount(actions, minlength=num_actions)
    probabilities = action_counts / len(actions)
    probabilities = probabilities[probabilities > 0]  # Remove zeros
    
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
    return entropy


def compare_agents(
    agent_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare performance of multiple agents.
    
    Args:
        agent_results: Dictionary mapping agent names to their results
        
    Returns:
        Comparison statistics
    """
    comparison = {}
    
    for agent_name, results in agent_results.items():
        comparison[agent_name] = {
            'mean_reward': results.get('mean_reward', 0),
            'std_reward': results.get('std_reward', 0),
            'mean_score': results.get('mean_score', 0),
            'convergence': results.get('convergence', {}),
        }
    
    # Find best agent
    best_agent = max(
        comparison.items(),
        key=lambda x: x[1]['mean_reward']
    )[0]
    
    comparison['best_agent'] = best_agent
    
    return comparison

