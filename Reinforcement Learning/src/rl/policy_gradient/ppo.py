"""
Proximal Policy Optimization (PPO) implementation for policy gradient methods.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional
from collections import deque
import pickle

from ..base.agent import RLAgent


class PPONetwork(nn.Module):
    """
    Neural network for PPO with shared feature extractor.
    """
    
    def __init__(self, state_dim: int, action_dim: int, continuous: bool = False):
        """
        Initialize PPO network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            continuous: Whether action space is continuous
        """
        super().__init__()
        self.continuous = continuous
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Policy head
        if continuous:
            self.policy_mean = nn.Linear(64, action_dim)
            self.policy_std = nn.Parameter(torch.ones(action_dim) * 0.5)
        else:
            self.policy = nn.Linear(64, action_dim)
        
        # Value head
        self.value = nn.Linear(64, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_logits/params: Policy output
            value: Value estimate
        """
        features = self.shared(state)
        
        if self.continuous:
            mean = self.policy_mean(features)
            std = torch.clamp(self.policy_std.exp(), min=1e-5)
            return mean, std
        else:
            action_logits = self.policy(features)
            return action_logits, self.value(features)


class PPOAgent(RLAgent):
    """
    Proximal Policy Optimization (PPO) agent.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        continuous: bool = False,
        **kwargs
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate
            discount_factor: Discount factor (gamma)
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
            update_epochs: Number of update epochs per batch
            batch_size: Batch size for updates
            continuous: Whether action space is continuous
        """
        super().__init__(state_dim, action_dim, **kwargs)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.continuous = continuous
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(state_dim, action_dim, continuous).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.reset_buffer()
    
    def reset_buffer(self):
        """Reset experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, **kwargs) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.continuous:
                mean, std = self.network(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                action = action.clamp(-1, 1)  # Assuming action space is [-1, 1]
                return action.cpu().numpy()[0], log_prob.item(), 0.0
            else:
                action_logits, value = self.network(state_tensor)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item(), value.item()
    
    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.network(state_tensor)
            return value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """Store transition in buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
        """
        advantages = np.zeros(len(rewards))
        returns = np.zeros(len(rewards))
        
        gae = 0
        next_value = next_value
        
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.discount_factor * next_value - values[step]
                gae = delta + self.discount_factor * self.gae_lambda * gae
            
            advantages[step] = gae
            returns[step] = gae + values[step]
            next_value = values[step]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, state: np.ndarray = None, action: int = None, 
               reward: float = None, next_state: np.ndarray = None, 
               done: bool = None, **kwargs):
        """
        Update policy using PPO algorithm.
        Note: PPO typically updates in batches, so this method
        should be called after collecting a batch of experiences.
        """
        if len(self.states) == 0:
            return
        
        # Get next value for GAE computation
        if not done and next_state is not None:
            next_value = self.get_value(next_state)
        else:
            next_value = 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.rewards, self.values, self.dones, next_value
        )
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Update for multiple epochs
        for epoch in range(self.update_epochs):
            # Shuffle data
            indices = torch.randperm(len(self.states))
            
            # Mini-batch updates
            for start in range(0, len(self.states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Forward pass
                if self.continuous:
                    mean, std = self.network(batch_states)
                    dist = Normal(mean, std)
                    action_samples = dist.sample()
                    new_log_probs = dist.log_prob(action_samples).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    action_logits, values = self.network(batch_states)
                    dist = Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    values = values.squeeze()
                
                # Compute policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                if not self.continuous:
                    value_loss = nn.functional.mse_loss(values, batch_returns)
                else:
                    _, values = self.network(batch_states)
                    values = values.squeeze()
                    value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Compute entropy bonus
                entropy_loss = -entropy
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Reset buffer
        self.reset_buffer()
    
    def save(self, filepath: str):
        """Save network and parameters."""
        save_data = {
            'network_state_dict': self.network.state_dict(),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'continuous': self.continuous,
        }
        torch.save(save_data, filepath)
    
    def load(self, filepath: str):
        """Load network and parameters."""
        save_data = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(save_data['network_state_dict'])
        self.learning_rate = save_data['learning_rate']
        self.discount_factor = save_data['discount_factor']
        self.state_dim = save_data['state_dim']
        self.action_dim = save_data['action_dim']
        self.continuous = save_data['continuous']

