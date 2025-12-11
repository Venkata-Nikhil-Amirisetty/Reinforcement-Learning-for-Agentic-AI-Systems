# Quick Start Guide

## Installation

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Option 1: Using the main script

```bash
# Run Q-Learning experiment
python main.py q_learning --episodes 1000

# Run PPO experiment
python main.py ppo --episodes 1000

# Run comparison
python main.py comparison --episodes 1000

# Run all experiments
python main.py all --episodes 1000
```

### Option 2: Using individual experiment scripts

```bash
# Q-Learning
python experiments/run_q_learning_experiment.py --episodes 1000

# PPO
python experiments/run_ppo_experiment.py --episodes 1000

# Comparison
python experiments/run_comparison.py --episodes 1000
```

## Quick Test

To quickly test if everything works:

```python
from src.agents.tutorial_agent import TutorialAgent

# Create agent
agent = TutorialAgent(rl_algorithm='q_learning')

# Train for a few episodes
agent.train(num_episodes=10, students_per_episode=2)

# Evaluate
results = agent.evaluate(num_students=5)
print(f"Mean Score: {results['mean_score']:.3f}")
```

## Understanding the Output

After running experiments, you'll find:

- **Results JSON**: `experiments/results/{algorithm}/results.json`
  - Training statistics
  - Evaluation metrics
  - Convergence analysis

- **Plots**: `experiments/results/{algorithm}/plots/`
  - Learning curves
  - Student performance over time
  - Action distributions

## Project Structure

```
.
├── src/                    # Source code
│   ├── rl/                # RL algorithms
│   ├── agents/            # Agent systems
│   └── utils/             # Utilities
├── experiments/           # Experiment scripts
├── docs/                  # Documentation
└── main.py               # Main entry point
```

## Next Steps

1. Review the technical report: `docs/technical_report.md`
2. Customize hyperparameters in experiment scripts
3. Modify reward functions in `src/agents/tutorial_agent.py`
4. Add your own evaluation metrics

## Troubleshooting

**Import errors**: Make sure you're in the project root directory and have activated the virtual environment.

**Memory issues**: Reduce `students_per_episode` or `num_episodes` for testing.

**Slow training**: This is normal for RL. Consider reducing episodes for initial testing.

