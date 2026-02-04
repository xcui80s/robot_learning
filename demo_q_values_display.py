"""
DEMO: Step-by-Step Robot Learning with Q-Value Display

This script demonstrates showing Q-values before and after learning
for the current state. Perfect for teaching Q-learning updates!

Usage:
    python demo_q_values_display.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.space_station import SpaceStationEnv
from agents.q_learning import QLearningAgent
from utils.visualizer import SpaceVisualizer
import numpy as np

def main():
    print("="*60)
    print("Q-VALUES BEFORE/AFTER LEARNING DEMO")
    print("="*60)
    print("\nThis demo shows Q-values for the current state:")
    print("- BEFORE learning (old values)")
    print("- AFTER learning (new values after update)")
    print("- CHANGE (delta showing how much each action improved)")
    print("\nWatch how the Q-value for the chosen action increases!")
    print("Press ESC to quit.\n")
    
    # Create environment
    env = SpaceStationEnv(grid_size=5, num_asteroids=2, num_energy=2)
    
    # Create agent
    agent = QLearningAgent(n_states=25, n_actions=4, epsilon=0.5)
    
    # Create visualizer
    viz = SpaceVisualizer(grid_size=5, fps=10)
    
    # Run one episode
    state, info = env.reset()
    viz.set_episode(1)
    viz.set_epsilon(agent.epsilon)
    
    print("Starting episode 1...")
    print("Watch the Q-values change as the robot learns!\n")
    
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 30:
        # Get Q-values BEFORE learning
        old_q_values = agent.get_q_values(state).copy()
        
        # Choose action
        action = agent.choose_action(state, training=True)
        
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Learn (this updates Q-values)
        agent.learn(state, action, reward, next_state, terminated)
        
        # Get Q-values AFTER learning
        new_q_values = agent.get_q_values(state).copy()
        
        # Calculate and display the changes
        changes = new_q_values - old_q_values
        print(f"\nStep {steps + 1}: State={state}, Action={action}, Reward={reward:.1f}")
        print(f"  BEFORE: {old_q_values.round(1)}")
        print(f"  AFTER:  {new_q_values.round(1)}")
        print(f"  CHANGE: {changes.round(1)}")
        print(f"  Action {action} changed by: {changes[action]:+.1f}")
        
        # Visualize with before/after Q-values
        viz.update(info, reward, old_q_values, new_q_values, state, action)
        
        total_reward += reward
        state = next_state
        steps += 1
    
    print(f"\nEpisode complete!")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.1f}")
    
    # Wait before closing
    import pygame
    pygame.time.wait(2000)
    viz.close()
    
    print("\nDemo complete!")
    print("\nKey observation: Notice how the Q-value for the action taken")
    print("usually increases after learning, especially when the reward was good!")

if __name__ == "__main__":
    main()
