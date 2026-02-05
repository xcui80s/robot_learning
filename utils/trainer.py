"""
🏋️ TRAINER UTILITIES 🏋️

This module helps train the robot with progress tracking,
visualization, and real-time explanations.

It ties together:
- The Space Station environment
- The Q-learning agent
- The visualizer (pygame)
- The concept explainer

Makes training easy and fun for kids to watch!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
from colorama import init, Fore, Style

from environments.space_station import SpaceStationEnv
from agents.q_learning import QLearningAgent
from utils.visualizer import SpaceVisualizer
from utils.concept_explainer import ConceptExplainer

init(autoreset=True)


class RobotTrainer:
    """
    🏋️ Trains the robot to navigate the space station!
    
    This class handles the entire training process:
    1. Creates the environment
    2. Creates the agent
    3. Runs training episodes
    4. Shows visualizations
    5. Explains what's happening
    6. Saves the trained robot
    
    Perfect for kids to watch reinforcement learning in action!
    """
    
    def __init__(
        self,
        grid_size: int = 5,
        num_asteroids: int = 3,
        num_energy: int = 2,
        learning_rate: float = 0.1,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        use_visualizer: bool = True,
        explain_mode: str = 'smart',
        fps: int = 10,
    ):
        """
        🎓 Set up the trainer.
        
        Args:
            grid_size: Size of the space station (5 = 5x5 grid)
            num_asteroids: Number of dangerous asteroids
            num_energy: Number of energy stars to collect
            learning_rate: How fast the robot learns
            epsilon_start: Initial exploration rate (1.0 = explore everything)
            epsilon_min: Minimum exploration rate
            epsilon_decay: How fast to reduce exploration
            use_visualizer: Show pygame visualization?
            explain_mode: 'verbose', 'smart', or 'minimal' explanations
            fps: Animation speed (frames per second)
        """
        # Create the space station
        self.env = SpaceStationEnv(
            grid_size=grid_size,
            num_asteroids=num_asteroids,
            num_energy=num_energy
        )
        
        # Create the robot brain
        n_states = grid_size * grid_size
        n_actions = 4  # UP, DOWN, LEFT, RIGHT
        
        self.agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=learning_rate,
            epsilon=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        )
        
        # Create visualizer (optional)
        self.visualizer = None
        if use_visualizer:
            self.visualizer = SpaceVisualizer(
                grid_size=grid_size,
                fps=fps
                # Uses default cell_size=100 for larger display
            )
        
        # Create concept explainer
        self.explainer = ConceptExplainer(
            mode=explain_mode,
            grid_size=grid_size
        )
        
        # Track training progress
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.best_score = float('-inf')
        
    def train(
        self,
        n_episodes: int = 500,
        max_steps_per_episode: int = 100,
        save_path: str = 'robot_brain.json',
        eval_interval: int = 100,
    ) -> Tuple[List[float], List[int]]:
        """
        🚀 Train the robot!
        
        This is the main training loop where the robot learns.
        
        Args:
            n_episodes: How many episodes to train for
            max_steps_per_episode: Max steps before episode ends
            save_path: Where to save the trained robot
            eval_interval: How often to evaluate progress
            
        Returns:
            episode_rewards: List of total reward per episode
            episode_lengths: List of steps per episode
        """
        print(f"\n{Fore.GREEN}🚀 Starting Training!{Style.RESET_ALL}")
        print(f"   Episodes: {n_episodes}")
        print(f"   Grid size: {self.env.grid_size}x{self.env.grid_size}")
        print(f"   Asteroids: {self.env.num_asteroids}")
        print(f"   Energy stars: {self.env.num_energy}\n")
        
        # Explain training start
        self.explainer.explain_training_start(n_episodes)
        
        # Training loop with progress bar
        pbar = tqdm(total=n_episodes, desc="Training", unit="episode")
        
        for episode in range(1, n_episodes + 1):
            # Reset environment
            state, info = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            # Set up visualization for this episode
            if self.visualizer:
                self.visualizer.set_episode(episode)
                self.visualizer.set_epsilon(self.agent.epsilon)
            
            # Explain episode start
            if episode <= 3 or episode % 50 == 0:
                self.explainer.explain_episode_start(episode, self.agent.epsilon)
            
            # Run one episode
            # Choose initial action (SARSA starts with an action)
            action, is_exploration = self.agent.choose_action(state, training=True)
            
            while not done and steps < max_steps_per_episode:
                # Get Q-values BEFORE learning
                old_q_values = self.agent.get_q_values(state).copy()
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Choose next_action from next_state
                next_action, next_is_exploration = self.agent.choose_action(next_state, training=True)
                
                # Learn from this experience (updates Q-values)
                # For SARSA: uses next_action to maintain trajectory consistency
                # For Q-learning: next_action is ignored, uses max(Q)
                self.agent.learn(state, action, reward, next_state, terminated, next_action)
                
                # Get Q-values AFTER learning
                new_q_values = self.agent.get_q_values(state).copy()
                
                # Update visualization with before/after Q-values
                if self.visualizer:
                    self.visualizer.update(info, reward, old_q_values, new_q_values, state, action, is_exploration)
                
                # Explain what happened
                self.explainer.explain_step(
                    env_info=info,
                    state=state,
                    action=action,
                    reward=reward,
                    q_values=new_q_values,  # Use updated Q-values after learning
                    training_mode=True
                )
                
                # Update tracking
                total_reward += reward
                
                # SARSA: Set current action = next_action for next iteration
                state = next_state
                action = next_action
                is_exploration = next_is_exploration
                steps += 1
            
            # Episode complete
            self.agent.episodes_completed += 1
            success = info.get('robot_pos') == self.env.goal_pos
            
            # Store results
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.successes.append(success)
            
            # Update best score
            if total_reward > self.best_score:
                self.best_score = total_reward
            
            # Update exploration rate
            self.agent.update_epsilon()
            
            # Explain episode end
            if episode <= 3 or episode % 50 == 0:
                self.explainer.explain_episode_end(
                    episode=episode,
                    total_reward=total_reward,
                    steps=steps,
                    success=success,
                    epsilon=self.agent.epsilon
                )
            
            # Progress update
            pbar.update(1)
            if episode % eval_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-eval_interval:])
                success_rate = np.mean(self.successes[-eval_interval:])
                pbar.set_postfix({
                    'avg_reward': f'{avg_reward:.1f}',
                    'success': f'{success_rate:.1%}',
                    'epsilon': f'{self.agent.epsilon:.3f}'
                })
        
        pbar.close()
        
        # Training complete
        self.explainer.explain_training_complete(n_episodes, self.best_score)
        
        # Print final summary
        self._print_training_summary()
        
        # Save the trained robot
        self.agent.save(save_path)
        
        return self.episode_rewards, self.episode_lengths
    
    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = True,
        slow_mode: bool = False,
    ) -> Tuple[float, float]:
        """
        🎮 Test the trained robot!
        
        Run the trained robot without exploration (only exploitation)
        to see how well it learned.
        
        Args:
            n_episodes: Number of test episodes
            render: Show visualization?
            slow_mode: Slow down for better viewing?
            
        Returns:
            avg_reward: Average reward across episodes
            success_rate: Percentage of successful episodes
        """
        print(f"\n{Fore.CYAN}🎮 Testing Trained Robot!{Style.RESET_ALL}")
        print(f"   Running {n_episodes} test episodes...")
        print(f"   (Robot will use what it learned, no random moves)\n")
        
        test_rewards = []
        test_successes = []
        
        for episode in range(1, n_episodes + 1):
            state, info = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            if self.visualizer:
                self.visualizer.set_episode(episode)
                self.visualizer.set_epsilon(0.0)  # No exploration
            
            print(f"Test Episode {episode}/{n_episodes}...")
            
            while not done and steps < 100:
                # Get Q-values (same before and after since we're not learning in test mode)
                q_values = self.agent.get_q_values(state).copy()
                
                # Choose best action (no exploration)
                action = self.agent.get_best_action(state)
                is_exploration = False  # Testing mode always uses exploitation
                
                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Render
                if render and self.visualizer:
                    # Pass same Q-values for before and after (no learning in test mode)
                    self.visualizer.update(info, reward, q_values, q_values, state, action, is_exploration)
                    if slow_mode:
                        import pygame
                        pygame.time.wait(500)  # Wait 500ms between steps
                
                total_reward += reward
                state = next_state
                steps += 1
            
            success = info.get('robot_pos') == self.env.goal_pos
            test_rewards.append(total_reward)
            test_successes.append(success)
            
            status = "✓" if success else "✗"
            print(f"  {status} Reward: {total_reward:.1f} | Steps: {steps}")
        
        avg_reward = np.mean(test_rewards)
        success_rate = np.mean(test_successes)
        
        print(f"\n{Fore.GREEN}Test Results:{Style.RESET_ALL}")
        print(f"  Average reward: {avg_reward:.1f}")
        print(f"  Success rate: {success_rate:.1%}")
        
        return avg_reward, success_rate
    
    def _print_training_summary(self):
        """
        📊 Print a summary of training results.
        """
        last_100_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        last_100_successes = self.successes[-100:] if len(self.successes) >= 100 else self.successes
        
        print(f"\n{Fore.GREEN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 TRAINING SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*50}{Style.RESET_ALL}")
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Best score: {self.best_score:.1f}")
        print(f"Average reward (last 100): {np.mean(last_100_rewards):.1f}")
        print(f"Success rate (last 100): {np.mean(last_100_successes):.1%}")
        print(f"Final epsilon: {self.agent.epsilon:.3f}")
        print(f"{Fore.GREEN}{'='*50}{Style.RESET_ALL}\n")
    
    def close(self):
        """
        🚪 Clean up resources.
        """
        if self.visualizer:
            self.visualizer.close()
        print(f"{Fore.CYAN}Trainer closed. Thanks for learning! 🚀{Style.RESET_ALL}")


# Helper function to quickly train a robot
def quick_train(
    grid_size: int = 5,
    n_episodes: int = 300,
    save_path: str = 'robot_brain.json',
    visualize: bool = True,
) -> RobotTrainer:
    """
    🚀 Quick function to train a robot with default settings.
    
    Args:
        grid_size: Size of grid (default: 5x5)
        n_episodes: How many episodes to train
        save_path: Where to save the brain
        visualize: Show pygame window?
        
    Returns:
        trainer: The trained RobotTrainer object
    """
    trainer = RobotTrainer(
        grid_size=grid_size,
        num_asteroids=3,
        num_energy=2,
        use_visualizer=visualize,
        explain_mode='smart',
        fps=10,
    )
    
    trainer.train(
        n_episodes=n_episodes,
        save_path=save_path
    )
    
    return trainer


# Test the trainer if we run this file directly
if __name__ == "__main__":
    print("🏋️ Testing Robot Trainer!")
    print("=" * 50)
    
    # Create a quick trainer
    trainer = RobotTrainer(
        grid_size=5,
        num_asteroids=2,
        num_energy=1,
        use_visualizer=False,  # Don't open window in test
        explain_mode='minimal',
    )
    
    # Train for a few episodes
    print("\n🚀 Quick training test (10 episodes)...")
    rewards, lengths = trainer.train(
        n_episodes=10,
        save_path='test_brain.json'
    )
    
    # Print agent summary
    print("\n🧠 Agent Summary:")
    trainer.agent.print_q_table_summary()
    
    # Clean up
    trainer.close()
    
    print("\n✅ Trainer test complete!")
