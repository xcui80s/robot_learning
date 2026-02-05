"""
🎯 LESSON 3: COLLECTION CHALLENGE - Fixed Layout! 🎯

Welcome to Lesson 3, Space Cadet! 👨‍🚀

In this lesson, you'll train a robot on a FIXED layout where it must:
1. Collect BOTH energy stars (⚡)
2. THEN reach the goal (⭐)

This teaches the robot MULTI-OBJECTIVE planning - it needs to:
- Remember the fixed layout
- Plan an efficient route
- Collect everything before going to the goal

THE CHALLENGE:
==============
This is a FIXED layout (never changes):

    0   1   2   3   4
  ┌───┬───┬───┬───┬───┐
0 │ 🤖│   │ ⚡│   │   │  <- Start + Energy #1 at (0,2)
  ├───┼───┼───┼───┼───┤
1 │   │   │   │ 💥│   │  <- Asteroid at (1,3)
  ├───┼───┼───┼───┼───┤
2 │   │ 💥│   │ ⚡│   │  <- Asteroid + Energy #2 at (2,1) and (2,3)
  ├───┼───┼───┼───┼───┤
3 │   │   │   │   │   │
  ├───┼───┼───┼───┼───┤
4 │   │   │   │   │ ⭐│  <- Goal at (4,4)
  └───┴───┴───┴───┴───┘

Rewards:
- Collect energy: +10 points each
- Collect ALL energy: +50 BONUS! 🎉
- Reach goal (with all energy): +100 points
- Reach goal (incomplete): +10 points
- Hit asteroid: -10 points
- Each step: -1 point

HOW TO USE:
===========
1. Run this file to train the robot:
   python lessons/03_collection_challenge/train_collection_robot.py

2. Then test your robot:
   python lessons/03_collection_challenge/play_collection_robot.py

EXPERIMENTS TO TRY:
===================
🔧 Change the layout (edit the FIXED_POSITIONS in collection_challenge.py)
🔧 Add more energy stars (try 3 or 4!)
🔧 Make it harder with more asteroids
🔧 Change the rewards to see different behavior
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environments.collection_challenge import CollectionChallengeEnv
from agents.q_learning import QLearningAgent
from utils.trainer import RobotTrainer
from utils.visualizer import SpaceVisualizer
from utils.concept_explainer import ConceptExplainer
from colorama import init, Fore, Style
import numpy as np

init(autoreset=True)


def train_collection_robot():
    """
    🎯 Train a robot on the Collection Challenge!
    """
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}🎯 LESSON 3: COLLECTION CHALLENGE!")
    print(f"{Fore.CYAN}{'='*60}\n")
    
    print(f"{Fore.WHITE}Welcome back, Space Cadet! 👨‍🚀")
    print(f"\nThis is a SPECIAL challenge with a FIXED layout!")
    print(f"The robot must collect ALL energy before reaching the goal.")
    
    print(f"\n{Fore.GREEN}📋 THE CHALLENGE:")
    print(f"  Grid: 5x5 fixed layout")
    print(f"  Energy: 2 stars to collect (positions: (0,2) and (2,3))")
    print(f"  Asteroids: 2 to avoid (positions: (1,3) and (2,1))")
    print(f"  Goal: Bottom-right corner (4,4)")
    print(f"  Objective: Collect BOTH stars, THEN reach goal")
    
    print(f"\n{Fore.YELLOW}🎮 PRESS ENTER TO START TRAINING!{Style.RESET_ALL}")
    input()
    
    # =========================================================================
    # 🔧 CONFIGURATION - You can change these!
    # =========================================================================
    
    GRID_SIZE = 5
    N_EPISODES = 400  # More episodes for this harder challenge!
    LEARNING_RATE = 0.1
    
    # Choose learning strategy
    print(f"\n{Fore.CYAN}🎯 Choose Learning Strategy:{Style.RESET_ALL}")
    print(f"  1. Q-Learning (off-policy) - Faster, more aggressive")
    print(f"  2. SARSA (on-policy) - Safer, more conservative")
    strategy_choice = input("  Enter choice (1 or 2, default=1): ").strip()
    
    if strategy_choice == '2':
        STRATEGY = 'sarsa'
        print(f"  ✓ Using SARSA strategy")
    else:
        STRATEGY = 'qlearning'
        print(f"  ✓ Using Q-Learning strategy")
    
    print(f"\n{Fore.CYAN}⚙️  Training Configuration:{Style.RESET_ALL}")
    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Episodes: {N_EPISODES}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Strategy: {STRATEGY.upper()}")
    print(f"  Mode: Fixed layout + Collection challenge")
    
    # Create the collection environment
    print(f"\n{Fore.GREEN}🎯 Creating Collection Challenge environment...")
    env = CollectionChallengeEnv(
        grid_size=GRID_SIZE,
        fixed_layout=True,
        collection_mode=True
    )
    
    # Show the layout
    print(f"\n{Fore.YELLOW}🗺️ Fixed Layout:{Style.RESET_ALL}")
    print(f"  Energy positions: {sorted(env.fixed_energy_positions)}")
    print(f"  Asteroid positions: {sorted(env.fixed_asteroid_positions)}")
    print(f"  Goal position: {env.goal_pos}")
    print(f"  Start position: {env.start_pos}")
    
    # Create the trainer
    print(f"{Fore.GREEN}\n🤖 Setting up robot trainer...")
    
    # We'll use a modified trainer that works with our collection env
    trainer = CollectionTrainer(
        env=env,
        use_visualizer=True,
        explain_mode='smart',
        fps=10,
        strategy=STRATEGY,
    )
    
    print(f"{Fore.GREEN}✅ Ready to train!\n")
    print(f"{Fore.YELLOW}🚀 Starting in 3 seconds...")
    print(f"{Fore.WHITE}Watch the robot learn the fixed layout!\n")
    
    import time
    time.sleep(3)
    
    # =========================================================================
    # 🚀 TRAIN THE ROBOT!
    # =========================================================================
    
    print(f"{Fore.GREEN}🏋️  Training the Collection Robot...\n")
    
    rewards, lengths = trainer.train(
        n_episodes=N_EPISODES,
        save_path='lessons/03_collection_challenge/collection_robot.json',
        eval_interval=50,
    )
    
    # =========================================================================
    # 🎉 RESULTS
    # =========================================================================
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}🎉 TRAINING COMPLETE!")
    print(f"{Fore.GREEN}{'='*60}\n")
    
    print(f"{Fore.WHITE}Your collection robot has been trained!")
    print(f"  Saved to: lessons/03_collection_challenge/collection_robot.json")
    
    print(f"\n{Fore.CYAN}📊 Results:{Style.RESET_ALL}")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Best score: {max(rewards):.1f} points")
    print(f"  Final average: {sum(rewards[-100:])/len(rewards[-100:]):.1f} points")
    
    # =========================================================================
    # 🎮 TEST THE ROBOT
    # =========================================================================
    
    print(f"\n{Fore.YELLOW}🎮 Now let's test your collection robot!{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Watch it collect both stars, then reach the goal.\n")
    print(f"{Fore.YELLOW}PRESS ENTER TO TEST!{Style.RESET_ALL}")
    input()
    
    test_results = trainer.test(
        n_episodes=5,
        render=True,
        slow_mode=True,
    )
    
    # =========================================================================
    # 🎓 WRAP UP
    # =========================================================================
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}🎓 LESSON 3 COMPLETE!")
    print(f"{Fore.GREEN}{'='*60}\n")
    
    print(f"{Fore.WHITE}Congratulations! You trained a multi-objective robot! 🎉")
    
    print(f"\n{Fore.CYAN}WHAT YOU LEARNED:{Style.RESET_ALL}")
    print(f"  ✅ Fixed layout memorization")
    print(f"  ✅ Multi-objective path planning")
    print(f"  ✅ Sequential task completion")
    print(f"  ✅ Collection strategy optimization")
    
    print(f"\n{Fore.YELLOW}🔬 EXPERIMENTS TO TRY:{Style.RESET_ALL}")
    print(f"  1. Edit collection_challenge.py and change energy positions")
    print(f"  2. Add more energy stars (try 3 or 4)")
    print(f"  3. Add more asteroids for a harder challenge")
    print(f"  4. Change the goal position")
    print(f"  5. Train for more episodes (try 600!)")
    
    print(f"\n{Fore.GREEN}🚀 WHAT'S NEXT?{Style.RESET_ALL}")
    print(f"  Run: python play_collection_robot.py")
    print(f"  To watch your robot solve the challenge anytime!")
    
    trainer.close()
    
    print(f"\n{Fore.CYAN}Great job, Space Cadet! 🚀{Style.RESET_ALL}\n")


class CollectionTrainer:
    """
    🎯 Special trainer for the Collection Challenge!
    
    Modified to work with CollectionChallengeEnv and track collection progress.
    """
    
    def __init__(self, env, use_visualizer=True, explain_mode='smart', fps=10, strategy='qlearning'):
        self.env = env
        self.use_visualizer = use_visualizer
        self.explain_mode = explain_mode
        self.fps = fps
        self.strategy = strategy
        
        # Create agent with chosen strategy
        n_states = env.grid_size * env.grid_size
        n_actions = 4
        
        self.agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=0.1,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            strategy=strategy,
        )
        
        # Create visualizer
        self.visualizer = None
        if use_visualizer:
            self.visualizer = SpaceVisualizer(
                grid_size=env.grid_size,
                fps=fps,
                pause_between_steps=False
                # Uses default cell_size=100 for larger display
            )
        
        # Create explainer
        self.explainer = ConceptExplainer(
            mode=explain_mode,
            grid_size=env.grid_size
        )
        
        # Track progress
        self.episode_rewards = []
        self.episode_lengths = []
        self.collection_successes = []  # Did it collect all energy?
        self.goal_successes = []  # Did it reach goal?
        self.best_score = float('-inf')
        
    def train(self, n_episodes=400, save_path='collection_robot.json', eval_interval=50):
        """
        🏋️ Train the collection robot!
        """
        from tqdm import tqdm
        
        print(f"{Fore.GREEN}🚀 Starting Collection Challenge Training!{Style.RESET_ALL}")
        print(f"   Episodes: {n_episodes}")
        print(f"   Grid: {self.env.grid_size}x{self.env.grid_size} (FIXED LAYOUT)")
        print(f"   Strategy: {self.strategy.upper()}")
        print(f"   Energy to collect: {self.env.total_energy_count}\n")
        
        # Training loop
        pbar = tqdm(total=n_episodes, desc="Training", unit="episode")
        
        for episode in range(1, n_episodes + 1):
            # Reset environment
            state, info = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            collected_all = False
            
            # Setup visualization
            if self.visualizer:
                self.visualizer.set_episode(episode)
                self.visualizer.set_epsilon(self.agent.epsilon)
            
            # Episode loop
            while not done and steps < 100:
                # Get Q-values BEFORE learning
                old_q_values = self.agent.get_q_values(state).copy()
                
                # Choose action
                action, is_exploration = self.agent.choose_action(state, training=True)
                
                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Learn (updates Q-values)
                self.agent.learn(state, action, reward, next_state, terminated)
                
                # Get Q-values AFTER learning
                new_q_values = self.agent.get_q_values(state).copy()
                
                # Visualize with before/after Q-values
                if self.visualizer:
                    self.visualizer.update(info, reward, old_q_values, new_q_values, state, action, is_exploration)
                
                # Track progress
                total_reward += reward
                state = next_state
                steps += 1
                
                # Check if collected all
                if info.get('collection_complete'):
                    collected_all = True
            
            # Episode complete
            self.agent.episodes_completed += 1
            reached_goal = info.get('robot_pos') == self.env.goal_pos
            mission_success = info.get('mission_success', False)
            
            # Store results
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.collection_successes.append(collected_all)
            self.goal_successes.append(reached_goal)
            
            if total_reward > self.best_score:
                self.best_score = total_reward
            
            # Update epsilon
            self.agent.update_epsilon()
            
            # Update progress bar
            pbar.update(1)
            if episode % eval_interval == 0:
                recent_rewards = self.episode_rewards[-eval_interval:]
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                success_rate = sum(self.goal_successes[-eval_interval:]) / eval_interval
                
                pbar.set_postfix({
                    'avg_reward': f'{avg_reward:.1f}',
                    'success': f'{success_rate:.1%}',
                    'epsilon': f'{self.agent.epsilon:.3f}'
                })
        
        pbar.close()
        
        # Save agent
        self.agent.save(save_path)
        
        # Print summary
        self._print_summary()
        
        return self.episode_rewards, self.episode_lengths
    
    def test(self, n_episodes=5, render=True, slow_mode=True):
        """
        🎮 Test the trained collection robot!
        """
        print(f"\n{Fore.CYAN}🎮 Testing Collection Robot!{Style.RESET_ALL}")
        print(f"   Running {n_episodes} test episodes...")
        print(f"   (Robot will use learned strategy, no random moves)\n")
        
        test_rewards = []
        test_collection = []
        test_goal = []
        
        for episode in range(1, n_episodes + 1):
            state, info = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            collected_all = False
            
            if self.visualizer:
                self.visualizer.set_episode(episode)
                self.visualizer.set_epsilon(0.0)
            
            print(f"Episode {episode}/{n_episodes}... ", end='')
            
            while not done and steps < 100:
                # Get Q-values (same before and after since we're not learning in test mode)
                q_values = self.agent.get_q_values(state).copy()
                action = self.agent.get_best_action(state)
                is_exploration = False  # Testing mode always uses exploitation
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                if render and self.visualizer:
                    # Pass same Q-values for before and after (no learning in test mode)
                    self.visualizer.update(info, reward, q_values, q_values, state, action, is_exploration)
                    if slow_mode:
                        import pygame
                        pygame.time.wait(500)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if info.get('collection_complete'):
                    collected_all = True
            
            reached_goal = info.get('robot_pos') == self.env.goal_pos
            
            test_rewards.append(total_reward)
            test_collection.append(collected_all)
            test_goal.append(reached_goal)
            
            status = "✓" if reached_goal and collected_all else "✗"
            print(f"{status} Reward: {total_reward:.1f} | Goal: {reached_goal} | Collected: {collected_all}")
        
        # Results
        print(f"\n{Fore.GREEN}Test Results:{Style.RESET_ALL}")
        print(f"  Average reward: {sum(test_rewards)/len(test_rewards):.1f}")
        print(f"  Collection rate: {sum(test_collection)/len(test_collection):.1%}")
        print(f"  Goal rate: {sum(test_goal)/len(test_goal):.1%}")
        print(f"  Perfect missions: {sum([c and g for c, g in zip(test_collection, test_goal)])}/{n_episodes}")
        
        return {
            'rewards': test_rewards,
            'collection': test_collection,
            'goal': test_goal
        }
    
    def _print_summary(self):
        """
        📊 Print training summary.
        """
        last_100 = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        
        print(f"\n{Fore.GREEN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 TRAINING SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*50}{Style.RESET_ALL}")
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Best score: {self.best_score:.1f}")
        print(f"Average reward (last 100): {sum(last_100)/len(last_100):.1f}")
        print(f"Collection success rate: {sum(self.collection_successes[-100:])/min(100, len(self.collection_successes)):.1%}")
        print(f"Goal success rate: {sum(self.goal_successes[-100:])/min(100, len(self.goal_successes)):.1%}")
        print(f"Final epsilon: {self.agent.epsilon:.3f}")
        print(f"{'='*50}\n")
    
    def close(self):
        """
        🚪 Clean up.
        """
        if self.visualizer:
            self.visualizer.close()


if __name__ == "__main__":
    try:
        train_collection_robot()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Training stopped. Progress saved!")
        print(f"You can resume training anytime by running this script again.")
