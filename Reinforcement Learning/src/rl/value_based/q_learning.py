"""
Q-Learning implementation for value-based reinforcement learning.
"""
import numpy as np
import pickle
from typing import Dict, Optional, Tuple
from collections import defaultdict
import random

from ..base.agent import RLAgent


class QLearningAgent(RLAgent):
    """
    Q-Learning agent for learning optimal action-value functions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        **kwargs
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum exploration rate
        """
        super().__init__(state_dim, action_dim, **kwargs)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> Q-value
        # Using dictionary for discrete states, can be extended to function approximation
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_dim)
        )
        
        # For continuous states, we'll use a discretization function
        self.state_discretizer = kwargs.get('state_discretizer', self._default_discretizer)
        
    def _default_discretizer(self, state: np.ndarray) -> Tuple:
        """
        Default state discretization for continuous states.
        Converts continuous state to discrete tuple representation.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Discretized state tuple
        """
        # Simple binning strategy - can be customized
        discretized = tuple(np.round(state * 10).astype(int))
        return discretized
    
    def _get_state_key(self, state: np.ndarray) -> Tuple:
        """Convert state to hashable key for Q-table."""
        return self.state_discretizer(state)
    
    def select_action(self, state: np.ndarray, **kwargs) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            **kwargs: Additional parameters (can override epsilon)
            
        Returns:
            Selected action index
        """
        epsilon = kwargs.get('epsilon', self.epsilon if self.training else 0.0)
        state_key = self._get_state_key(state)
        
        # Exploration: random action
        if self.training and random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Exploitation: best action according to Q-table
        q_values = self.q_table[state_key]
        # Handle ties randomly
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool, **kwargs):
        """
        Update Q-values using Q-Learning update rule.
        
        Q(s,a) = Q(s,a) + α[r + γ * max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is complete
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Next state max Q-value
        if done:
            next_max_q = 0.0
        else:
            next_max_q = np.max(self.q_table[next_state_key])
        
        # Q-Learning update
        target_q = reward + self.discount_factor * next_max_q
        td_error = target_q - current_q
        new_q = current_q + self.learning_rate * td_error
        
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for state-action pair."""
        state_key = self._get_state_key(state)
        return self.q_table[state_key][action]
    
    def get_state_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions in given state."""
        state_key = self._get_state_key(state)
        return self.q_table[state_key].copy()
    
    def save(self, filepath: str):
        """Save Q-table and parameters."""
        save_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, filepath: str):
        """Load Q-table and parameters."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_table = defaultdict(
            lambda: np.zeros(save_data['action_dim']),
            save_data['q_table']
        )
        self.epsilon = save_data['epsilon']
        self.learning_rate = save_data['learning_rate']
        self.discount_factor = save_data['discount_factor']
        self.state_dim = save_data['state_dim']
        self.action_dim = save_data['action_dim']


class DQNAgent(RLAgent):
    """
    Deep Q-Network (DQN) agent using neural network function approximation.
    This is an advanced version that can handle continuous state spaces better.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        **kwargs
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate for neural network
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum exploration rate
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch")
        
        super().__init__(state_dim, action_dim, **kwargs)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Replay buffer
        self.memory = []
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def _build_network(self):
        """Build the Q-network architecture."""
        import torch.nn as nn
        
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def select_action(self, state: np.ndarray, **kwargs) -> int:
        """Select action using epsilon-greedy policy."""
        import torch
        
        epsilon = kwargs.get('epsilon', self.epsilon if self.training else 0.0)
        
        if self.training and random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def _store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool, **kwargs):
        """Update Q-network using experience replay."""
        import torch
        
        self._store_transition(state, action, reward, next_state, done)
        
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.discount_factor * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save Q-network and parameters."""
        import torch
        
        save_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }
        torch.save(save_data, filepath)
    
    def load(self, filepath: str):
        """Load Q-network and parameters."""
        import torch
        
        save_data = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(save_data['q_network_state_dict'])
        self.target_network.load_state_dict(save_data['q_network_state_dict'])
        self.epsilon = save_data['epsilon']
        self.learning_rate = save_data['learning_rate']
        self.discount_factor = save_data['discount_factor']

