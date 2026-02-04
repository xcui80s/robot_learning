"""
🎮 PLAY WITH YOUR TRAINED ROBOT! 🎮

This script lets you watch your trained robot anytime!

Run this after you've trained a robot with train_robot.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.q_learning import QLearningAgent
from environments.space_station import SpaceStationEnv
from utils.visualizer import SpaceVisualizer
from colorama import init, Fore, Style

init(autoreset=True)


def main():
    """
    🎮 Watch the trained robot play!
    """
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}🎮 PLAYING WITH TRAINED ROBOT!")
    print(f"{Fore.CYAN}{'='*60}\n")
    
    # Check if robot brain exists
    brain_path = 'lessons/02_space_adventure/robot_brain.json'
    
    if not os.path.exists(brain_path):
        print(f"{Fore.RED}❌ Robot brain not found!")
        print(f"{Fore.WHITE}You need to train a robot first.")
        print(f"\nRun: python lessons/02_space_adventure/train_robot.py")
        return
    
    # Load the trained robot
    print(f"{Fore.GREEN}🧠 Loading robot brain...")
    agent = QLearningAgent.load(brain_path)
    
    # Create the environment
    print(f"{Fore.GREEN}🚀 Creating space station...")
    env = SpaceStationEnv(grid_size=5, num_asteroids=3, num_energy=2)
    
    # Create visualizer
    print(f"{Fore.GREEN}🎨 Starting visualizer...\n")
    viz = SpaceVisualizer(grid_size=5, fps=5)  # Uses default cell_size=100 for larger display
    
    # Play some episodes
    n_episodes = 5
    print(f"{Fore.YELLOW}🎮 Watching {n_episodes} episodes...")
    print(f"{Fore.WHITE}(Press ESC or close window to stop)\n")
    
    total_rewards = []
    successes = []
    
    for episode in range(1, n_episodes + 1):
        # Reset environment
        state, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        viz.set_episode(episode)
        viz.set_epsilon(0.0)  # No exploration - just use what it learned
        
        print(f"Episode {episode}/{n_episodes}... ", end='')
        
        while not done and steps < 100:
            # Get Q-values (same before and after since we're not learning in play mode)
            q_values = agent.get_q_values(state).copy()
            
            # Choose best action (no randomness)
            action = agent.get_best_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Visualize - pass same Q-values for before and after (no learning in play mode)
            viz.update(info, reward, q_values, q_values, state, action, False)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        # Check if succeeded
        success = info.get('robot_pos') == env.goal_pos
        total_rewards.append(total_reward)
        successes.append(success)
        
        status = "✓" if success else "✗"
        print(f"{status} Reward: {total_reward:.1f}")
    
    # Print summary
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}📊 RESULTS")
    print(f"{Fore.GREEN}{'='*60}")
    print(f"Average reward: {sum(total_rewards)/len(total_rewards):.1f}")
    print(f"Success rate: {sum(successes)/len(successes):.1%}")
    print(f"Best reward: {max(total_rewards):.1f}")
    print(f"{'='*60}\n")
    
    # Clean up
    viz.close()
    
    print(f"{Fore.CYAN}Thanks for playing! 🚀{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
