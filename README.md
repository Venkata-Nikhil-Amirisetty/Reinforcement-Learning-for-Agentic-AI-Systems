# Reinforcement Learning for Agentic AI Systems

This project implements reinforcement learning mechanisms to enhance the performance of agentic AI systems, specifically focusing on adaptive tutorial agents that learn optimal teaching strategies through student interactions.

## Project Overview

This implementation integrates two reinforcement learning approaches:
1. **Q-Learning (Value-Based)**: For learning optimal action sequences in tutorial interactions
2. **PPO (Policy Gradient)**: For optimizing teaching policy based on student feedback

The system is designed to work with tutorial agent frameworks similar to Humanitarians.AI's Dewey framework, learning to:
- Optimize question sequences and difficulty progression
- Adapt scaffolding strategies based on learner performance
- Personalize teaching approaches through reinforcement learning

## Project Structure

```
.
├── src/
│   ├── rl/
│   │   ├── value_based/
│   │   │   └── q_learning.py
│   │   ├── policy_gradient/
│   │   │   └── ppo.py
│   │   └── base/
│   │       └── agent.py
│   ├── agents/
│   │   ├── tutorial_agent.py
│   │   └── orchestration.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── Architecture diagrams/
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.png
│   └── 5.png
├── experiments/
│   ├── run_q_learning_experiment.py
│   ├── run_ppo_experiment.py
│   ├── run_comparison.py
│   ├── regenerate_all_visualizations.py
│   └── results/
├── docs/
│   └── Reinforcement Learning for Agentic AI Systems.pdf
├── main.py
├── test_basic.py
├── requirements.txt
└── setup.py
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.agents.tutorial_agent import TutorialAgent
from src.rl.value_based.q_learning import QLearningAgent
from src.rl.policy_gradient.ppo import PPOAgent

# Initialize tutorial agent with Q-Learning
agent = TutorialAgent(rl_algorithm='q_learning')

# Train the agent
agent.train(num_episodes=1000)

# Evaluate performance
results = agent.evaluate()
```

## Key Features

- **Dual RL Implementation**: Both value-based (Q-Learning, DQN) and policy gradient (PPO) methods
- **Adaptive Learning**: Agents improve through student interaction feedback
- **Comprehensive Evaluation**: Metrics for learning performance and agent behavior
- **Visualization Tools**: Learning curves, policy visualization, and behavior analysis
- **Production-Ready**: Well-structured codebase with proper documentation

## Project Highlights

This project fulfills the requirements for a reinforcement learning course final project:

✅ **Two RL Approaches Implemented**:
   - Q-Learning (Value-Based) with tabular and DQN variants
   - PPO (Policy Gradient) with actor-critic architecture

✅ **Agentic System Integration**:
   - Adaptive tutorial agent system (similar to Humanitarians.AI's Dewey framework)
   - Learning optimal question sequences and difficulty progression
   - Personalized scaffolding strategies

✅ **Comprehensive Evaluation**:
   - Learning curves and convergence analysis
   - Statistical validation of results
   - Comparative analysis between algorithms

✅ **Complete Documentation**:
   - Technical report with mathematical formulations
   - System architecture documentation
   - Code documentation and examples

## Experiments

Run experiments from the `experiments/` directory:

```bash
python experiments/run_q_learning_experiment.py
python experiments/run_ppo_experiment.py
python experiments/run_comparison.py
```

## Documentation

- **Technical Report**: `docs/Reinforcement Learning for Agentic AI Systems.pdf`
- **Architecture Diagrams**: `Architecture diagrams/` (5 diagrams)
- **Visualization Guide**: `VISUALIZATION_GUIDE.md`
- **Quick Start Guide**: `QUICKSTART.md`
- **API Documentation**: See docstrings in source files

## License

This project is for educational purposes as part of a reinforcement learning course.

