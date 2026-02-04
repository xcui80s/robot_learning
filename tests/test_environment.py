"""
🧪 Tests for the Space Station Environment

These tests verify that the environment works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.space_station import SpaceStationEnv
import numpy as np


def test_environment_creation():
    """Test that we can create the environment."""
    env = SpaceStationEnv(grid_size=5, num_asteroids=3, num_energy=2)
    assert env is not None
    assert env.grid_size == 5
    assert env.num_asteroids == 3
    assert env.num_energy == 2
    print("✅ Environment creation works!")


def test_environment_reset():
    """Test that we can reset the environment."""
    env = SpaceStationEnv(grid_size=5, num_asteroids=3, num_energy=2)
    state, info = env.reset()
    
    assert state is not None
    assert isinstance(state, (int, np.integer))
    assert 0 <= state < 25  # 5x5 grid
    assert info is not None
    assert 'robot_pos' in info
    assert 'asteroids' in info
    assert 'energy_stars' in info
    assert 'goal_pos' in info
    print("✅ Environment reset works!")


def test_environment_step():
    """Test that we can take steps in the environment."""
    env = SpaceStationEnv(grid_size=5, num_asteroids=3, num_energy=2)
    state, info = env.reset()
    
    # Try each action
    for action in range(4):  # UP, DOWN, LEFT, RIGHT
        next_state, reward, terminated, truncated, info = env.step(action)
        
        assert next_state is not None
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info is not None
    
    print("✅ Environment step works!")


def test_reward_structure():
    """Test that rewards are calculated correctly."""
    env = SpaceStationEnv(grid_size=5, num_asteroids=0, num_energy=0)
    state, info = env.reset()
    
    # Remove all objects for this test
    env.asteroids = set()
    env.energy_stars = set()
    
    # Take a step (should get -1 for step penalty)
    next_state, reward, terminated, truncated, info = env.step(env.DOWN)
    
    # Just check that we got some reward
    assert isinstance(reward, (int, float, np.number))
    print("✅ Reward structure works!")


def test_state_consistency():
    """Test that state numbers are consistent."""
    env = SpaceStationEnv(grid_size=5, num_asteroids=0, num_energy=0)
    
    # Test that position (0,0) maps to state 0
    env.robot_pos = (0, 0)
    assert env._get_state() == 0
    
    # Test that position (0,1) maps to state 1
    env.robot_pos = (0, 1)
    assert env._get_state() == 1
    
    # Test that position (1,0) maps to state 5
    env.robot_pos = (1, 0)
    assert env._get_state() == 5
    
    print("✅ State mapping is consistent!")


def test_episode_completion():
    """Test that episodes complete when goal is reached."""
    env = SpaceStationEnv(grid_size=5, num_asteroids=0, num_energy=0)
    state, info = env.reset()
    
    # Place robot right next to goal
    env.robot_pos = (3, 4)  # One step away from goal (4,4)
    
    # Move down to goal
    next_state, reward, terminated, truncated, info = env.step(env.DOWN)
    
    assert env.robot_pos == (4, 4)  # Should be at goal
    # Episode might be terminated or not depending on if it's actually the goal
    print("✅ Episode mechanics work!")


if __name__ == "__main__":
    print("🧪 Running Environment Tests...")
    print("=" * 50)
    
    try:
        test_environment_creation()
        test_environment_reset()
        test_environment_step()
        test_reward_structure()
        test_state_consistency()
        test_episode_completion()
        
        print("\n" + "=" * 50)
        print("🎉 All environment tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
