"""
Base classes for reinforcement learning agents.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class RLAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    """
    
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        """
        Initialize the RL agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            **kwargs: Additional agent-specific parameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training = True
        
    @abstractmethod
    def select_action(self, state: np.ndarray, **kwargs) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            **kwargs: Additional parameters (e.g., epsilon for exploration)
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool, **kwargs):
        """
        Update the agent's policy/value function based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is complete
            **kwargs: Additional parameters
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save the agent's model/parameters."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load the agent's model/parameters."""
        pass
    
    def train(self):
        """Set agent to training mode."""
        self.training = True
    
    def eval(self):
        """Set agent to evaluation mode."""
        self.training = False


class StateSpace:
    """
    Represents the state space for tutorial agent interactions.
    """
    
    def __init__(self, features: Dict[str, int]):
        """
        Initialize state space.
        
        Args:
            features: Dictionary mapping feature names to their dimensions
        """
        self.features = features
        self.dim = sum(features.values())
    
    def encode(self, **kwargs) -> np.ndarray:
        """
        Encode state features into a vector.
        
        Args:
            **kwargs: State feature values
            
        Returns:
            Encoded state vector
        """
        state = []
        for feature_name, dim in self.features.items():
            value = kwargs.get(feature_name, 0)
            if dim == 1:
                state.append(float(value))
            else:
                # One-hot encoding for categorical features
                encoding = np.zeros(dim)
                if isinstance(value, int) and 0 <= value < dim:
                    encoding[value] = 1.0
                state.extend(encoding)
        return np.array(state, dtype=np.float32)


class ActionSpace:
    """
    Represents the action space for tutorial agent.
    """
    
    def __init__(self, actions: list):
        """
        Initialize action space.
        
        Args:
            actions: List of possible actions
        """
        self.actions = actions
        self.n = len(actions)
        self.action_to_idx = {action: idx for idx, action in enumerate(actions)}
        self.idx_to_action = {idx: action for idx, action in enumerate(actions)}
    
    def get_action(self, idx: int):
        """Get action from index."""
        return self.idx_to_action.get(idx)
    
    def get_idx(self, action):
        """Get index from action."""
        return self.action_to_idx.get(action)

