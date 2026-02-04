"""
DEMO: Step-by-Step Robot Learning with Pause

This script demonstrates the pause feature that waits for user input
between each robot action. Perfect for teaching and explaining each step!

Usage:
    python demo_pause_mode.py

Press SPACE or Click to advance to the next step.
Press ESC to quit.
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
    print("STEP-BY-STEP ROBOT LEARNING DEMO")
    print("="*60)
    print("\nThis demo shows the robot learning with PAUSE between each action.")
    print("Press SPACE or Click to continue to the next step.")
    print("Press ESC to quit.\n")
    
    # Create environment
    env = SpaceStationEnv(grid_size=5, num_asteroids=2, num_energy=2)
    
    # Create agent
    agent = QLearningAgent(n_states=25, n_actions=4, epsilon=0.5)
    
    # Create visualizer with PAUSE MODE enabled
    print("Creating visualizer with pause_between_steps=True...")
    viz = SpaceVisualizer(
        grid_size=5, 
        fps=30,
        pause_between_steps=True  # This enables the pause feature!
    )
    
    # Run one episode with pauses
    state, info = env.reset()
    viz.set_episode(1)
    viz.set_epsilon(agent.epsilon)
    
    print("Starting episode 1...")
    print("Watch the robot and press SPACE or Click to advance each step.\n")
    
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 50:
        # Get Q-values BEFORE learning (old values)
        old_q_values = agent.get_q_values(state).copy()
        
        # Choose action
        action = agent.choose_action(state, training=True)
        
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Learn (this updates Q-values)
        agent.learn(state, action, reward, next_state, terminated)
        
        # Get Q-values AFTER learning (new values)
        new_q_values = agent.get_q_values(state).copy()
        
        # Visualize with PAUSE - show before/after Q-values
        viz.update(info, reward, old_q_values, new_q_values, state, action)
        
        total_reward += reward
        state = next_state
        steps += 1
        
        print(f"Step {steps}: Action={action}, Reward={reward:.1f}, Total={total_reward:.1f}")
    
    print(f"\nEpisode complete!")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.1f}")
    
    # Wait a bit before closing
    import pygame
    pygame.time.wait(2000)
    viz.close()
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
