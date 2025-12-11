"""
Adaptive Tutorial Agent with Reinforcement Learning.
This agent learns optimal teaching strategies through student interactions.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from ..rl.base.agent import RLAgent, StateSpace, ActionSpace
from ..rl.value_based.q_learning import QLearningAgent, DQNAgent
from ..rl.policy_gradient.ppo import PPOAgent


class QuestionType(Enum):
    """Types of questions the agent can ask."""
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    CONCEPTUAL = "conceptual"
    PRACTICE = "practice"
    REVIEW = "review"


class DifficultyLevel(Enum):
    """Difficulty levels for questions."""
    EASY = 0
    MEDIUM = 1
    HARD = 2


class ScaffoldingStrategy(Enum):
    """Teaching scaffolding strategies."""
    DIRECT_INSTRUCTION = "direct_instruction"
    GUIDED_DISCOVERY = "guided_discovery"
    SCAFFOLDED_PRACTICE = "scaffolded_practice"
    INDEPENDENT_PRACTICE = "independent_practice"
    HINT_PROVIDED = "hint_provided"


class StudentResponse:
    """Represents a student's response to a question."""
    
    def __init__(
        self,
        correct: bool,
        time_taken: float,
        attempts: int = 1,
        confidence: Optional[float] = None
    ):
        self.correct = correct
        self.time_taken = time_taken
        self.attempts = attempts
        self.confidence = confidence


class TutorialAgent:
    """
    Adaptive tutorial agent that uses reinforcement learning to optimize
    teaching strategies based on student interactions.
    """
    
    def __init__(
        self,
        rl_algorithm: str = 'q_learning',
        use_dqn: bool = False,
        **rl_kwargs
    ):
        """
        Initialize tutorial agent.
        
        Args:
            rl_algorithm: RL algorithm to use ('q_learning', 'dqn', or 'ppo')
            use_dqn: Whether to use DQN instead of tabular Q-learning
            **rl_kwargs: Additional arguments for RL agent
        """
        self.rl_algorithm = rl_algorithm
        
        # Define state space: [student_level, topic_mastery, recent_performance, 
        #                      question_count, time_spent, difficulty_preference]
        self.state_space = StateSpace({
            'student_level': 5,  # 0-4: beginner to advanced
            'topic_mastery': 10,  # 0-9: mastery levels
            'recent_performance': 5,  # 0-4: performance bins
            'question_count': 10,  # 0-9: question count bins
            'time_spent': 10,  # 0-9: time bins
            'difficulty_preference': 3,  # 0-2: easy, medium, hard
        })
        
        # Define action space: combinations of question type, difficulty, and scaffolding
        self.action_space = ActionSpace([
            (qt, d, s) for qt in QuestionType 
            for d in DifficultyLevel 
            for s in ScaffoldingStrategy
        ])
        
        # Initialize RL agent
        if rl_algorithm == 'ppo':
            self.rl_agent: RLAgent = PPOAgent(
                state_dim=self.state_space.dim,
                action_dim=self.action_space.n,
                continuous=False,  # Discrete action space
                **rl_kwargs
            )
        elif rl_algorithm == 'dqn' or use_dqn:
            self.rl_agent = DQNAgent(
                state_dim=self.state_space.dim,
                action_dim=self.action_space.n,
                **rl_kwargs
            )
        else:
            self.rl_agent = QLearningAgent(
                state_dim=self.state_space.dim,
                action_dim=self.action_space.n,
                **rl_kwargs
            )
        
        # Agent state
        self.current_state = None
        self.last_action_data = None  # For PPO: stores (log_prob, value)
        self.student_profile = {
            'level': 0,  # Beginner
            'topic_mastery': {},
            'recent_responses': [],
            'question_count': 0,
            'total_time': 0.0,
        }
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'student_scores': [],
            'learning_curves': [],
        }
    
    def encode_state(self) -> np.ndarray:
        """
        Encode current state from student profile.
        
        Returns:
            Encoded state vector
        """
        # Calculate average topic mastery
        if self.student_profile['topic_mastery']:
            avg_mastery = np.mean(list(self.student_profile['topic_mastery'].values()))
        else:
            avg_mastery = 0.0
        
        # Calculate recent performance (last 5 responses)
        recent = self.student_profile['recent_responses'][-5:]
        if recent:
            recent_perf = np.mean([1.0 if r.correct else 0.0 for r in recent])
        else:
            recent_perf = 0.5  # Neutral
        
        # Normalize values
        student_level = min(4, max(0, int(self.student_profile['level'])))
        mastery_bin = min(9, max(0, int(avg_mastery * 9)))
        perf_bin = min(4, max(0, int(recent_perf * 4)))
        q_count_bin = min(9, max(0, self.student_profile['question_count'] // 5))
        time_bin = min(9, max(0, int(self.student_profile['total_time'] / 60)))  # minutes
        
        # Difficulty preference based on recent performance
        if recent_perf > 0.8:
            diff_pref = 2  # Prefer harder
        elif recent_perf < 0.4:
            diff_pref = 0  # Prefer easier
        else:
            diff_pref = 1  # Medium
        
        return self.state_space.encode(
            student_level=student_level,
            topic_mastery=mastery_bin,
            recent_performance=perf_bin,
            question_count=q_count_bin,
            time_spent=time_bin,
            difficulty_preference=diff_pref
        )
    
    def compute_reward(
        self,
        action: Tuple,
        response: StudentResponse,
        previous_state: np.ndarray
    ) -> float:
        """
        Compute reward based on student response and teaching effectiveness.
        
        Args:
            action: Action taken (question_type, difficulty, scaffolding)
            response: Student's response
            previous_state: Previous state
            
        Returns:
            Reward value
        """
        question_type, difficulty, scaffolding = action
        
        # Base reward for correctness
        if response.correct:
            base_reward = 1.0
        else:
            base_reward = -0.5
        
        # Efficiency bonus: faster correct answers are better
        if response.correct:
            time_bonus = max(0, 1.0 - response.time_taken / 60.0) * 0.3
        else:
            time_bonus = 0
        
        # Difficulty matching: appropriate difficulty gets bonus
        student_level = self.student_profile['level']
        diff_level = difficulty.value
        
        if abs(diff_level - student_level) <= 1:
            difficulty_bonus = 0.2
        elif diff_level > student_level + 1:
            difficulty_bonus = -0.3  # Too hard
        else:
            difficulty_bonus = -0.1  # Too easy
        
        # Scaffolding effectiveness: hints should help but not be overused
        if scaffolding == ScaffoldingStrategy.HINT_PROVIDED:
            if response.correct and response.attempts <= 2:
                scaffolding_bonus = 0.1
            else:
                scaffolding_bonus = -0.1
        else:
            scaffolding_bonus = 0.0
        
        # Learning progress: improvement over time
        recent = self.student_profile['recent_responses'][-10:]
        if len(recent) >= 5:
            old_perf = np.mean([1.0 if r.correct else 0.0 for r in recent[:5]])
            new_perf = np.mean([1.0 if r.correct else 0.0 for r in recent[-5:]])
            progress_bonus = (new_perf - old_perf) * 0.5
        else:
            progress_bonus = 0.0
        
        total_reward = base_reward + time_bonus + difficulty_bonus + scaffolding_bonus + progress_bonus
        
        return total_reward
    
    def select_action(self, state: np.ndarray) -> Tuple:
        """
        Select teaching action using RL agent.
        
        Args:
            state: Current state
            
        Returns:
            Selected action (question_type, difficulty, scaffolding)
        """
        if self.rl_algorithm == 'ppo':
            action_idx, log_prob, value = self.rl_agent.select_action(state)
            self.last_action_data = (log_prob, value)
        else:
            action_idx = self.rl_agent.select_action(state)
            self.last_action_data = None
        
        return self.action_space.get_action(action_idx)
    
    def update_student_profile(self, response: StudentResponse, topic: str = "default"):
        """Update student profile based on response."""
        self.student_profile['question_count'] += 1
        self.student_profile['total_time'] += response.time_taken
        self.student_profile['recent_responses'].append(response)
        
        # Keep only last 20 responses
        if len(self.student_profile['recent_responses']) > 20:
            self.student_profile['recent_responses'].pop(0)
        
        # Update topic mastery
        if topic not in self.student_profile['topic_mastery']:
            self.student_profile['topic_mastery'][topic] = 0.0
        
        if response.correct:
            self.student_profile['topic_mastery'][topic] = min(
                1.0,
                self.student_profile['topic_mastery'][topic] + 0.1
            )
            # Increase level if consistently performing well
            if len(self.student_profile['recent_responses']) >= 5:
                recent_correct = sum(1 for r in self.student_profile['recent_responses'][-5:] if r.correct)
                if recent_correct >= 4:
                    self.student_profile['level'] = min(4, self.student_profile['level'] + 0.1)
        else:
            self.student_profile['topic_mastery'][topic] = max(
                0.0,
                self.student_profile['topic_mastery'][topic] - 0.05
            )
    
    def interact(self, topic: str = "default") -> Tuple:
        """
        Interact with student: select action and return it.
        
        Args:
            topic: Current topic being taught
            
        Returns:
            Selected action (question_type, difficulty, scaffolding)
        """
        state = self.encode_state()
        self.current_state = state
        action = self.select_action(state)
        return action
    
    def receive_feedback(
        self,
        action: Tuple,
        response: StudentResponse,
        topic: str = "default"
    ):
        """
        Receive student feedback and update agent.
        
        Args:
            action: Action that was taken
            response: Student's response
            topic: Topic being taught
        """
        previous_state = self.current_state
        reward = self.compute_reward(action, response, previous_state)
        
        # Update student profile
        self.update_student_profile(response, topic)
        
        # Get new state
        new_state = self.encode_state()
        
        # Update RL agent
        action_idx = self.action_space.get_idx(action)
        
        if self.rl_algorithm == 'ppo':
            # PPO collects experiences in buffer
            if self.last_action_data is not None:
                log_prob, value = self.last_action_data
                self.rl_agent.store_transition(
                    previous_state, action_idx, reward, log_prob, value, False
                )
        else:
            # Q-Learning/DQN update
            self.rl_agent.update(
                previous_state, action_idx, reward, new_state, False
            )
        
        self.current_state = new_state
    
    def train(self, num_episodes: int = 1000, students_per_episode: int = 10):
        """
        Train the agent through simulated student interactions.
        
        Args:
            num_episodes: Number of training episodes
            students_per_episode: Number of student interactions per episode
        """
        self.rl_agent.train()
        
        for episode in range(num_episodes):
            episode_rewards = []
            episode_scores = []
            
            # Simulate interactions with multiple students
            for student_idx in range(students_per_episode):
                # Reset student profile for new student
                self.student_profile = {
                    'level': np.random.uniform(0, 2),  # Random initial level
                    'topic_mastery': {},
                    'recent_responses': [],
                    'question_count': 0,
                    'total_time': 0.0,
                }
                
                student_rewards = []
                student_correct = 0
                
                # Simulate session with questions
                for q in range(20):  # 20 questions per student
                    action = self.interact()
                    
                    # Simulate student response
                    # Probability of correct answer depends on difficulty and student level
                    question_type, difficulty, scaffolding = action
                    diff_level = difficulty.value
                    student_level = self.student_profile['level']
                    
                    # Base correctness probability
                    base_prob = 0.5 + (student_level - diff_level) * 0.2
                    base_prob = np.clip(base_prob, 0.1, 0.9)
                    
                    # Scaffolding helps
                    if scaffolding == ScaffoldingStrategy.HINT_PROVIDED:
                        base_prob += 0.2
                    elif scaffolding == ScaffoldingStrategy.GUIDED_DISCOVERY:
                        base_prob += 0.1
                    
                    correct = np.random.random() < base_prob
                    time_taken = np.random.uniform(10, 120)  # 10-120 seconds
                    attempts = 1 if correct else np.random.randint(2, 4)
                    
                    response = StudentResponse(
                        correct=correct,
                        time_taken=time_taken,
                        attempts=attempts
                    )
                    
                    self.receive_feedback(action, response)
                    
                    reward = self.compute_reward(action, response, self.current_state)
                    student_rewards.append(reward)
                    if correct:
                        student_correct += 1
                
                episode_rewards.extend(student_rewards)
                episode_scores.append(student_correct / 20.0)
            
            # Update PPO if using it
            if self.rl_algorithm == 'ppo':
                self.rl_agent.update()
            
            # Record statistics
            avg_reward = float(np.mean(episode_rewards))
            avg_score = float(np.mean(episode_scores))
            
            self.training_stats['episodes'].append(episode)
            self.training_stats['rewards'].append(avg_reward)
            self.training_stats['student_scores'].append(avg_score)
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.3f}, Avg Score: {avg_score:.3f}")
    
    def evaluate(self, num_students: int = 50) -> Dict[str, Any]:
        """
        Evaluate agent performance.
        
        Args:
            num_students: Number of test students
            
        Returns:
            Evaluation metrics
        """
        self.rl_agent.eval()
        
        all_scores = []
        all_times = []
        difficulty_usage = {d: 0 for d in DifficultyLevel}
        scaffolding_usage = {s: 0 for s in ScaffoldingStrategy}
        
        for student_idx in range(num_students):
            # Reset student profile
            self.student_profile = {
                'level': np.random.uniform(0, 4),
                'topic_mastery': {},
                'recent_responses': [],
                'question_count': 0,
                'total_time': 0.0,
            }
            
            student_correct = 0
            student_time = 0.0
            
            for q in range(20):
                action = self.interact()
                question_type, difficulty, scaffolding = action
                
                difficulty_usage[difficulty] += 1
                scaffolding_usage[scaffolding] += 1
                
                # Simulate response
                diff_level = difficulty.value
                student_level = self.student_profile['level']
                base_prob = 0.5 + (student_level - diff_level) * 0.2
                base_prob = np.clip(base_prob, 0.1, 0.9)
                
                if scaffolding == ScaffoldingStrategy.HINT_PROVIDED:
                    base_prob += 0.2
                
                correct = np.random.random() < base_prob
                time_taken = np.random.uniform(10, 120)
                
                response = StudentResponse(correct=correct, time_taken=time_taken)
                self.update_student_profile(response)
                
                if correct:
                    student_correct += 1
                student_time += time_taken
            
            all_scores.append(student_correct / 20.0)
            all_times.append(student_time)
        
        return {
            'mean_score': float(np.mean(all_scores)),
            'std_score': float(np.std(all_scores)),
            'mean_time': float(np.mean(all_times)),
            'difficulty_distribution': {k.name: int(v) for k, v in difficulty_usage.items()},
            'scaffolding_distribution': {k.name: int(v) for k, v in scaffolding_usage.items()},
        }
    
    def save(self, filepath: str):
        """Save agent state."""
        self.rl_agent.save(filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        self.rl_agent.load(filepath)

