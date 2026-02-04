"""
🎮 PLAY COLLECTION CHALLENGE! 🎮

Watch your trained collection robot solve the fixed layout challenge!

Run this after training with train_collection_robot.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environments.collection_challenge import CollectionChallengeEnv
from agents.q_learning import QLearningAgent
from utils.visualizer import SpaceVisualizer
from colorama import init, Fore, Style

init(autoreset=True)


def main():
    """
    🎮 Play with the trained collection robot!
    """
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}🎮 PLAYING COLLECTION CHALLENGE!")
    print(f"{Fore.CYAN}{'='*60}\n")
    
    # Check if robot brain exists
    brain_path = 'lessons/03_collection_challenge/collection_robot.json'
    
    if not os.path.exists(brain_path):
        print(f"{Fore.RED}❌ Robot brain not found!")
        print(f"{Fore.WHITE}You need to train a collection robot first.")
        print(f"\nRun: python lessons/03_collection_challenge/train_collection_robot.py")
        return
    
    # Load the trained robot
    print(f"{Fore.GREEN}🧠 Loading collection robot brain...")
    agent = QLearningAgent.load(brain_path)
    
    # Create the environment
    print(f"{Fore.GREEN}🎯 Creating collection challenge environment...")
    env = CollectionChallengeEnv(
        grid_size=5,
        fixed_layout=True,
        collection_mode=True
    )
    
    # Show the layout
    print(f"\n{Fore.YELLOW}🗺️ Fixed Layout:{Style.RESET_ALL}")
    print(f"  Energy positions: {sorted(env.fixed_energy_positions)}")
    print(f"  Asteroid positions: {sorted(env.fixed_asteroid_positions)}")
    print(f"  Goal position: {env.goal_pos}")
    print(f"\n{Fore.CYAN}Mission: Collect BOTH energy stars, then reach the goal!\n")
    
    # Create visualizer
    print(f"{Fore.GREEN}🎨 Starting visualizer...\n")
    viz = SpaceVisualizer(grid_size=5, fps=5)  # Uses default cell_size=100 for larger display
    
    # Play some episodes
    n_episodes = 5
    print(f"{Fore.YELLOW}🎮 Watching {n_episodes} episodes...")
    print(f"{Fore.WHITE}(Press ESC or close window to stop)\n")
    
    total_rewards = []
    collection_success = []
    goal_success = []
    perfect_missions = []
    
    for episode in range(1, n_episodes + 1):
        # Reset environment
        state, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        collected_all = False
        
        viz.set_episode(episode)
        viz.set_epsilon(0.0)  # No exploration - use learned strategy
        
        print(f"Episode {episode}/{n_episodes}... ", end='', flush=True)
        
        while not done and steps < 100:
            # Get Q-values for visualization
            q_values = agent.get_q_values(state)
            
            # Choose best action (no randomness)
            action = agent.get_best_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Visualize
            viz.update(info, reward, q_values, state)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            # Track if collected all energy
            if info.get('collection_complete'):
                collected_all = True
        
        # Check results
        reached_goal = info.get('robot_pos') == env.goal_pos
        mission_complete = collected_all and reached_goal
        
        total_rewards.append(total_reward)
        collection_success.append(collected_all)
        goal_success.append(reached_goal)
        perfect_missions.append(mission_complete)
        
        # Print status
        if mission_complete:
            status = "✓ PERFECT!"
            color = Fore.GREEN
        elif reached_goal:
            status = "✓ Goal reached"
            color = Fore.YELLOW
        else:
            status = "✗ Failed"
            color = Fore.RED
        
        print(f"{color}{status} | Reward: {total_reward:.1f} | Steps: {steps}")
    
    # Print summary
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}📊 RESULTS")
    print(f"{Fore.GREEN}{'='*60}")
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    collection_rate = sum(collection_success) / len(collection_success)
    goal_rate = sum(goal_success) / len(goal_success)
    perfect_rate = sum(perfect_missions) / len(perfect_missions)
    
    print(f"{Fore.WHITE}Average reward: {avg_reward:.1f}")
    print(f"{Fore.WHITE}Collection rate: {collection_rate:.1%} (collected all energy)")
    print(f"{Fore.WHITE}Goal rate: {goal_rate:.1%} (reached the goal)")
    print(f"{Fore.GREEN}Perfect missions: {perfect_rate:.1%} (collected all + reached goal)")
    print(f"{Fore.GREEN}Best reward: {max(total_rewards):.1f}")
    print(f"{'='*60}\n")
    
    # Provide feedback
    if perfect_rate >= 0.8:
        print(f"{Fore.GREEN}🎉 EXCELLENT! Your robot is a collection expert!")
    elif perfect_rate >= 0.5:
        print(f"{Fore.YELLOW}👍 GOOD! Your robot usually completes the mission.")
    else:
        print(f"{Fore.CYAN}💡 Your robot is learning. Try training for more episodes!")
    
    # Clean up
    viz.close()
    
    print(f"\n{Fore.CYAN}Thanks for playing the Collection Challenge! 🚀{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
