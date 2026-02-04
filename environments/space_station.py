"""
🚀 SPACE STATION ENVIRONMENT 🚀

This is where our robot learns to navigate!
It's like a game board where the robot moves around collecting stars
and avoiding asteroids.

The robot learns by trying different moves and remembering what works!
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import random


class SpaceStationEnv(gym.Env):
    """
    🎮 A space station where a robot learns to navigate!
    
    The robot starts at one corner and needs to reach the goal (star).
    Along the way, it can collect energy and must avoid asteroids.
    
    STATE (what the robot knows):
    - Where it is (row, column)
    - How much energy it has collected
    - How many steps it has taken
    
    ACTIONS (what the robot can do):
    - 0: UP ⬆️
    - 1: DOWN ⬇️
    - 2: LEFT ⬅️
    - 3: RIGHT ➡️
    
    REWARDS (points for good/bad moves):
    - Reach the goal: +100 points! 🎉
    - Collect energy: +10 points ⭐
    - Hit asteroid: -10 points 💥
    - Each step: -1 point (encourages finding shortest path)
    """
    
    # These are our action names - makes code easier to read!
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    # Rewards - change these to experiment!
    REWARD_GOAL = 100
    REWARD_ENERGY = 10
    REWARD_ASTEROID = -10
    REWARD_STEP = -1
    
    def __init__(self, grid_size: int = 5, num_asteroids: int = 3, num_energy: int = 2):
        """
        🤖 Set up the space station!
        
        Args:
            grid_size: How big is our space station? (default: 5x5)
            num_asteroids: How many dangerous asteroids? (default: 3)
            num_energy: How many energy stars to collect? (default: 2)
        """
        super().__init__()
        
        # Save our settings
        self.grid_size = grid_size
        self.num_asteroids = num_asteroids
        self.num_energy = num_energy
        
        # The robot's starting position (top-left corner)
        self.start_pos = (0, 0)
        
        # The goal position (bottom-right corner)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        
        # Action space: 4 directions (UP, DOWN, LEFT, RIGHT)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: where is the robot?
        # We use a simple flat representation: row * grid_size + column
        # For a 5x5 grid, this gives us positions 0-24
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        
        # These will be set when we reset the environment
        self.robot_pos = None
        self.asteroids = set()
        self.energy_stars = set()
        self.collected_energy = 0
        self.steps = 0
        self.max_steps = grid_size * grid_size * 2  # Don't wander forever!
        
    def _get_state(self) -> int:
        """
        🗺️ Convert robot position to a state number.
        
        The state tells us WHERE the robot is in the grid.
        For a 5x5 grid:
        - Position (0,0) = state 0
        - Position (0,1) = state 1
        - Position (1,0) = state 5
        - etc.
        
        This helps the robot remember what to do at each spot!
        """
        row, col = self.robot_pos
        return row * self.grid_size + col
    
    def _place_objects(self):
        """
        🎲 Randomly place asteroids and energy stars.
        
        We make sure not to place them on:
        - The robot's starting position
        - The goal position
        - Where another object already is
        """
        self.asteroids = set()
        self.energy_stars = set()
        
        # All possible positions
        all_positions = [
            (r, c) for r in range(self.grid_size) 
            for c in range(self.grid_size)
        ]
        
        # Remove start and goal positions
        available = [pos for pos in all_positions 
                    if pos not in [self.start_pos, self.goal_pos]]
        
        # Place asteroids (dangerous!)
        for _ in range(self.num_asteroids):
            if available:
                pos = random.choice(available)
                self.asteroids.add(pos)
                available.remove(pos)
        
        # Place energy stars (collect them!)
        for _ in range(self.num_energy):
            if available:
                pos = random.choice(available)
                self.energy_stars.add(pos)
                available.remove(pos)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[int, Dict]:
        """
        🔄 Start a new episode!
        
        This puts the robot back at the start and creates a new space station layout.
        
        Returns:
            state: The starting state number
            info: Extra information about the environment
        """
        super().reset(seed=seed)
        
        # Reset robot to starting position
        self.robot_pos = self.start_pos
        
        # Reset counters
        self.collected_energy = 0
        self.steps = 0
        
        # Create a new random layout
        self._place_objects()
        
        # Return the starting state and some info
        state = self._get_state()
        info = {
            'robot_pos': self.robot_pos,
            'asteroids': list(self.asteroids),
            'energy_stars': list(self.energy_stars),
            'goal_pos': self.goal_pos,
            'collected_energy': self.collected_energy
        }
        
        return state, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        🎮 Make a move and see what happens!
        
        This is the main function that:
        1. Moves the robot based on the action
        2. Checks what the robot hit
        3. Calculates the reward
        4. Checks if the episode is done
        
        Args:
            action: Which direction to move (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            state: New position as a state number
            reward: Points gained or lost
            terminated: Did we reach the goal or crash?
            truncated: Did we take too many steps?
            info: Extra information for the robot
        """
        # Count this step
        self.steps += 1
        
        # Get current position (before move)
        old_row, old_col = self.robot_pos
        
        # Calculate distance to goal BEFORE moving (Manhattan distance)
        goal_row, goal_col = self.goal_pos
        old_distance = abs(old_row - goal_row) + abs(old_col - goal_col)
        
        # Try to move the robot
        if action == self.UP:
            new_pos = (old_row - 1, old_col)
        elif action == self.DOWN:
            new_pos = (old_row + 1, old_col)
        elif action == self.LEFT:
            new_pos = (old_row, old_col - 1)
        elif action == self.RIGHT:
            new_pos = (old_row, old_col + 1)
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0-3.")
        
        # Check if the move is valid (inside the grid)
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            self.robot_pos = new_pos
        # If invalid, robot stays in place (bumps into wall)
        
        # Calculate distance to goal AFTER moving
        new_row, new_col = self.robot_pos
        new_distance = abs(new_row - goal_row) + abs(new_col - goal_col)
        
        # Distance-based reward: +5 if closer, -5 if farther
        distance_reward = 0
        if new_distance < old_distance:
            distance_reward = 5  # Moving closer to goal - good!
        elif new_distance > old_distance:
            distance_reward = -5  # Moving away from goal - bad!
        # If distance is same (hit wall or moving perpendicular), no bonus
        
        # Now let's see what happened at this new position!
        reward = self.REWARD_STEP  # Small penalty for each step
        reward += distance_reward  # Add distance-based reward
        terminated = False
        truncated = False
        
        # Check for energy star
        if self.robot_pos in self.energy_stars:
            reward += self.REWARD_ENERGY
            self.energy_stars.remove(self.robot_pos)
            self.collected_energy += 1
        
        # Check for asteroid (OUCH!)
        if self.robot_pos in self.asteroids:
            reward += self.REWARD_ASTEROID
            terminated = True  # Episode ends if we crash!
        
        # Check for goal (VICTORY!)
        if self.robot_pos == self.goal_pos:
            reward += self.REWARD_GOAL
            terminated = True  # Episode ends when we succeed!
        
        # Check if we took too many steps
        if self.steps >= self.max_steps:
            truncated = True
        
        # Prepare info for next step
        info = {
            'robot_pos': self.robot_pos,
            'asteroids': list(self.asteroids),
            'energy_stars': list(self.energy_stars),
            'goal_pos': self.goal_pos,
            'collected_energy': self.collected_energy,
            'steps': self.steps,
            'action': action  # Remember what action we took
        }
        
        return self._get_state(), reward, terminated, truncated, info
    
    def render(self):
        """
        👀 Draw the space station (for debugging in terminal).
        
        This prints a text version of the grid.
        """
        print("\n" + "=" * (self.grid_size * 4 + 1))
        
        for row in range(self.grid_size):
            row_str = "|"
            for col in range(self.grid_size):
                pos = (row, col)
                
                if pos == self.robot_pos:
                    row_str += " 🤖|"
                elif pos == self.goal_pos:
                    row_str += " ⭐|"
                elif pos in self.asteroids:
                    row_str += " 💥|"
                elif pos in self.energy_stars:
                    row_str += " ⚡|"
                else:
                    row_str += "   |"
            
            print(row_str)
            print("=" * (self.grid_size * 4 + 1))
        
        print(f"Energy collected: {self.collected_energy}")
        print(f"Steps taken: {self.steps}")


# Test the environment if we run this file directly
if __name__ == "__main__":
    print("🚀 Testing Space Station Environment!")
    print("=" * 50)
    
    # Create the environment
    env = SpaceStationEnv(grid_size=5, num_asteroids=3, num_energy=2)
    
    # Start a new episode
    state, info = env.reset()
    print(f"\n🤖 Robot starts at position: {info['robot_pos']}")
    print(f"🎯 Goal is at position: {info['goal_pos']}")
    print(f"💥 Asteroids at: {info['asteroids']}")
    print(f"⚡ Energy stars at: {info['energy_stars']}")
    
    # Show the initial state
    env.render()
    
    # Try some random moves
    print("\n🎮 Testing random moves...")
    for i in range(5):
        action = env.action_space.sample()
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        print(f"\n➡️  Move {i+1}: Going {action_names[action]}")
        
        state, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        print(f"📊 Reward: {reward} points")
        
        if terminated:
            print("🎉 Episode finished! (goal reached or crashed)")
            break
        if truncated:
            print("⏰ Too many steps!")
            break
    
    print("\n✅ Environment test complete!")
