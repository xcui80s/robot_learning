"""
🔧 COLLECTION CHALLENGE - Fixed Layout Extension

This module extends the SpaceStationEnv for Lesson 3:
- Fixed layout mode (same positions every time)
- Collection challenge (must collect all energy before goal)
- Custom asteroid and energy positions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.space_station import SpaceStationEnv
from typing import Optional, Dict, Tuple, List, Set


class CollectionChallengeEnv(SpaceStationEnv):
    """
    🎯 COLLECTION CHALLENGE MODE!
    
    Same as SpaceStationEnv but with:
    - Fixed layout (you control where everything is!)
    - Collection mode (must collect ALL energy before goal)
    
    Perfect for teaching multi-objective path planning!
    
    FIXED LAYOUT (5x5):
    ====================
        0   1   2   3   4
      ┌───┬───┬───┬───┬───┐
    0 │ 🤖│   │ ⚡│   │   │  <- Start + First Energy at (0,2)
      ├───┼───┼───┼───┼───┤
    1 │   │   │   │ 💥│   │  <- Asteroid at (1,3)
      ├───┼───┼───┼───┼───┤
    2 │   │ 💥│   │ ⚡│   │  <- Asteroid + Energy at (2,1) and (2,3)
      ├───┼───┼───┼───┼───┤
    3 │   │   │   │   │   │
      ├───┼───┼───┼───┼───┤
    4 │   │   │   │   │ ⭐│  <- Goal at (4,4)
      └───┴───┴───┴───┴───┘
    
    Objective: Collect BOTH energy stars (⚡), then reach goal (⭐)
    
    Rewards:
    - Collect energy: +10 points
    - Collect ALL energy: +50 bonus!
    - Reach goal (with all energy): +100 points
    - Reach goal (without all energy): +10 points
    - Hit asteroid: -10 points
    - Each step: -1 point
    """
    
    # Fixed positions for Lesson 3
    FIXED_ENERGY_POSITIONS = [(0, 2), (2, 3)]  # Two energy stars
    FIXED_ASTEROID_POSITIONS = [(1, 3), (2, 1)]  # Two asteroids
    
    def __init__(
        self, 
        grid_size: int = 5,
        fixed_layout: bool = True,
        collection_mode: bool = True,
        custom_energy_positions: Optional[List[Tuple[int, int]]] = None,
        custom_asteroid_positions: Optional[List[Tuple[int, int]]] = None
    ):
        """
        🎯 Create Collection Challenge Environment!
        
        Args:
            grid_size: Size of the grid (default 5x5)
            fixed_layout: Use fixed positions? (True = same layout every time)
            collection_mode: Must collect all energy before goal bonus?
            custom_energy_positions: Override default energy positions
            custom_asteroid_positions: Override default asteroid positions
        """
        # Initialize parent with 0 random objects (we'll place them manually)
        super().__init__(grid_size=grid_size, num_asteroids=0, num_energy=0)
        
        self.fixed_layout = fixed_layout
        self.collection_mode = collection_mode
        
        # Store fixed positions
        if custom_energy_positions is not None:
            self.fixed_energy_positions = set(custom_energy_positions)
        else:
            self.fixed_energy_positions = set(self.FIXED_ENERGY_POSITIONS)
        
        if custom_asteroid_positions is not None:
            self.fixed_asteroid_positions = set(custom_asteroid_positions)
        else:
            self.fixed_asteroid_positions = set(self.FIXED_ASTEROID_POSITIONS)
        
        # Track collection
        self.total_energy_count = len(self.fixed_energy_positions)
        self.initial_energy_positions = set()  # Will be set in reset
        
    def _place_objects(self):
        """
        📍 Place objects in fixed positions (instead of random).
        """
        if self.fixed_layout:
            # Use fixed positions
            self.asteroids = self.fixed_asteroid_positions.copy()
            self.energy_stars = self.fixed_energy_positions.copy()
            self.initial_energy_positions = self.fixed_energy_positions.copy()
            self.total_energy_count = len(self.fixed_energy_positions)
        else:
            # Use parent's random placement
            super()._place_objects()
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        🎮 Make a move with collection challenge logic!
        
        Special rules:
        - Must collect ALL energy before getting full goal reward
        - Shows collection progress
        - Bonus for collecting all energy
        """
        # Count energy before move
        energy_before = len(self.initial_energy_positions) - len(self.energy_stars)
        
        # Call parent's step
        state, reward, terminated, truncated, info = super().step(action)
        
        # Count energy after move
        energy_after = len(self.initial_energy_positions) - len(self.energy_stars)
        
        # Check if we just collected the last energy
        if energy_after == len(self.initial_energy_positions) and energy_before < energy_after:
            # Just collected the last energy! Give bonus!
            reward += 50  # BONUS for collecting all!
            info['collection_complete'] = True
            info['bonus_earned'] = 50
        else:
            info['collection_complete'] = False
            info['bonus_earned'] = 0
        
        # Special goal logic for collection mode
        if self.collection_mode and self.robot_pos == self.goal_pos:
            # Check if we collected all energy
            if len(self.energy_stars) == 0:
                # SUCCESS! Collected all and reached goal
                info['mission_success'] = True
                info['completion_message'] = "🎉 MISSION COMPLETE! All energy collected + Goal reached!"
            else:
                # Reached goal but didn't collect all
                info['mission_success'] = False
                info['completion_message'] = "⚠️ Reached goal but didn't collect all energy!"
                # Reduce reward since we didn't complete the mission
                reward = 10  # Partial credit
        
        # Add collection info
        info['energy_collected'] = energy_after
        info['energy_total'] = len(self.initial_energy_positions)
        info['energy_remaining'] = len(self.energy_stars)
        
        return state, reward, terminated, truncated, info
    
    def render(self):
        """
        👀 Draw the space station with collection info.
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
        
        # Show collection progress
        collected = len(self.initial_energy_positions) - len(self.energy_stars)
        total = len(self.initial_energy_positions)
        print(f"Energy: {collected}/{total} collected")
        print(f"Steps: {self.steps}")
        
        if collected == total:
            print("✅ ALL ENERGY COLLECTED! Now reach the goal!")


# Test the collection environment
if __name__ == "__main__":
    print("🎯 Testing Collection Challenge Environment!")
    print("=" * 50)
    
    # Create the environment
    env = CollectionChallengeEnv(grid_size=5)
    
    print("\n🗺️ Fixed Layout:")
    print(f"Energy positions: {sorted(env.fixed_energy_positions)}")
    print(f"Asteroid positions: {sorted(env.fixed_asteroid_positions)}")
    print(f"Goal position: {env.goal_pos}")
    
    # Start episode
    state, info = env.reset()
    print(f"\n🤖 Robot starts at position: {info['robot_pos']}")
    print(f"🎯 Goal: Collect ALL {info['energy_total']} energy stars, then reach goal!")
    
    # Show initial state
    env.render()
    
    # Test collecting energy
    print("\n🎮 Test: Moving to collect first energy star...")
    
    # Move RIGHT twice to get energy at (0,2)
    for action in [3, 3]:  # RIGHT, RIGHT
        state, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Reward: {reward} points")
        
        if info.get('energy_remaining') == 0:
            print("🎉 ALL ENERGY COLLECTED! Bonus earned!")
            break
    
    print("\n✅ Collection Challenge Environment works!")
    print("Ready for Lesson 3!")
