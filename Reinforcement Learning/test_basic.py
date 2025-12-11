"""
Basic test to verify the system works.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.tutorial_agent import TutorialAgent, StudentResponse


def test_q_learning():
    """Test Q-Learning agent."""
    print("Testing Q-Learning agent...")
    agent = TutorialAgent(rl_algorithm='q_learning')
    
    # Simulate a few interactions
    for i in range(5):
        action = agent.interact()
        print(f"  Question {i+1}: {action[0].name}, {action[1].name}, {action[2].name}")
        
        # Simulate response
        correct = i % 2 == 0  # Alternate correct/incorrect
        response = StudentResponse(correct=correct, time_taken=30.0)
        agent.receive_feedback(action, response)
    
    print("✓ Q-Learning test passed\n")


def test_ppo():
    """Test PPO agent."""
    print("Testing PPO agent...")
    agent = TutorialAgent(rl_algorithm='ppo')
    
    # Simulate a few interactions
    for i in range(5):
        action = agent.interact()
        print(f"  Question {i+1}: {action[0].name}, {action[1].name}, {action[2].name}")
        
        # Simulate response
        correct = i % 2 == 0
        response = StudentResponse(correct=correct, time_taken=30.0)
        agent.receive_feedback(action, response)
    
    # Trigger PPO update
    agent.rl_agent.update()
    print("✓ PPO test passed\n")


def test_training():
    """Test training for a few episodes."""
    print("Testing training (short run)...")
    agent = TutorialAgent(rl_algorithm='q_learning')
    agent.train(num_episodes=5, students_per_episode=2)
    
    # Evaluate
    results = agent.evaluate(num_students=3)
    print(f"  Evaluation score: {results['mean_score']:.3f}")
    print("✓ Training test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Basic System Tests")
    print("=" * 60)
    print()
    
    try:
        test_q_learning()
        test_ppo()
        test_training()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

