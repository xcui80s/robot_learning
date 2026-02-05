"""
🧠 Q-LEARNING ROBOT BRAIN 🧠

This is where the magic happens! 
The Q-learning algorithm helps our robot learn which moves are best.

WHAT IS Q-LEARNING?
====================
Imagine you're learning to ride a bike:
- You try something (like pedaling)
- You see what happens (you move forward!)
- You remember: "Pedaling = good!"
- Next time, you're more likely to pedal

Q-learning does the same thing with MATH!

THE Q-TABLE (ROBOT'S MEMORY)
============================
The robot has a big table called the "Q-table" where it stores memories.
For each position in the space station, it remembers:
- If I go UP from here, how good was that?
- If I go DOWN from here, how good was that?
- If I go LEFT from here, how good was that?
- If I go RIGHT from here, how good was that?

The "Q" stands for "Quality" - how good is each move?
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import random


class QLearningAgent:
    """
    🤖 A robot that learns using Q-learning!
    
    The robot starts knowing NOTHING. It learns by:
    1. Trying random moves (exploration)
    2. Remembering which moves gave good rewards
    3. Using what it learned (exploitation)
    
    KEY IDEAS:
    =========
    - State: Where the robot is (like a position on a map)
    - Action: What the robot does (UP, DOWN, LEFT, RIGHT)
    - Reward: Points for good/bad moves
    - Q-Table: The robot's memory (state → action quality)
    """
    
    def __init__(
        self,
        n_states: int,      # How many different positions?
        n_actions: int,     # How many different moves? (4 for us)
        learning_rate: float = 0.1,    # How fast does it learn?
        discount_factor: float = 0.95,  # How much does it care about future rewards?
        epsilon: float = 1.0,           # How much does it explore vs use what it knows?
        epsilon_decay: float = 0.995,   # Does it explore less over time?
        epsilon_min: float = 0.01,      # Minimum exploration (never stop completely)
        strategy: str = 'qlearning',    # Learning strategy: 'qlearning' or 'sarsa'
    ):
        """
        🎓 Create a new robot brain!
        
        Args:
            n_states: Number of possible positions in the grid
            n_actions: Number of possible moves (UP, DOWN, LEFT, RIGHT = 4)
            learning_rate: How much we update our memory (0.1 = small steps)
            discount_factor: How much we care about future rewards (0.95 = care a lot)
            epsilon: Chance of exploring vs using knowledge (1.0 = explore everything)
            epsilon_decay: How fast we stop exploring (0.995 = slow transition)
            epsilon_min: Never stop exploring completely (0.01 = 1% random moves)
            strategy: Learning algorithm - 'qlearning' (off-policy) or 'sarsa' (on-policy)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.strategy = strategy
        
        # 🤖 THE Q-TABLE: Robot's memory!
        # It's a big table with zeros at first.
        # Rows = positions (states)
        # Columns = moves (actions: UP, DOWN, LEFT, RIGHT)
        # Values = how good each move is from that position
        self.q_table = np.zeros((n_states, n_actions))
        
        # Keep track of what we've learned
        self.total_reward = 0
        self.episodes_completed = 0
        
    def choose_action(self, state: int, training: bool = True) -> Tuple[int, bool]:
        """
        🎲 Choose what move to make!
        
        The robot has two modes:
        
        EXPLORATION (when training and epsilon is high):
        - Try random moves to discover new things
        - Like trying different paths in a maze
        
        EXPLOITATION (when epsilon is low or not training):
        - Use what we already learned
        - Pick the move with the highest Q-value
        
        Args:
            state: Current position as a number
            training: Are we learning or just playing?
            
        Returns:
            Tuple of (action, is_exploration)
            - action: Which move to make (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            - is_exploration: True if action was randomly chosen (exploration),
                               False if best action was chosen (exploitation)
        """
        # 🎲 EXPLORATION: Try something random!
        # The epsilon value is like a "curiosity level"
        # High epsilon = very curious, try new things
        # Low epsilon = confident, use what we know
        if training and random.random() < self.epsilon:
            # Pick a completely random move!
            # This helps us discover new strategies
            is_exploration = True
            action = random.randint(0, self.n_actions - 1)
            return action, is_exploration
        
        # 🧠 EXPLOITATION: Use what we learned!
        # Look at the Q-table for this position
        # Pick the move with the highest score
        is_exploration = False
        q_values = self.q_table[state]
        
        # Find the best move (the one with highest Q-value)
        best_action = int(np.argmax(q_values))
        
        # Sometimes multiple moves have the same best score
        # In that case, pick randomly among the best ones
        best_q_value = q_values[best_action]
        best_actions = [int(x) for x in np.where(q_values == best_q_value)[0]]
        
        if len(best_actions) > 1:
            # Multiple best moves! Pick one randomly
            action = random.choice(best_actions)
        else:
            action = best_action
            
        return int(action), is_exploration
    
    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        🎓 LEARN FROM EXPERIENCE!
        
        This is the heart of Q-learning. The robot updates its memory based on:
        - Where it was (state)
        - What it did (action)
        - What happened (reward)
        - Where it ended up (next_state)
        - Is the game over? (done)
        
        THE Q-LEARNING FORMULA:
        ======================
        Think of it like updating a score:
        
        New Score = Old Score + Learning Rate × (Reward + Future Value - Old Score)
        
        Or in math:
        Q(state, action) = Q(state, action) + α × (reward + γ × max(Q(next_state)) - Q(state, action))
        
        WHERE:
        - Q(state, action) = How good was this move? (starts at 0)
        - α (alpha) = Learning rate (how fast we learn)
        - reward = Points we got right now
        - γ (gamma) = Discount factor (how much we care about future rewards)
        - max(Q(next_state)) = Best possible move from the NEW position
        
        Args:
            state: Where we were
            action: What we did
            reward: Points we got
            next_state: Where we ended up
            done: Is the episode finished?
        """
        # Step 1: Remember the old score for this move
        old_value = self.q_table[state, action]
        
        # Step 2: Calculate the "future value"
        # If we're done (reached goal or crashed), there's no future
        # Otherwise, look ahead: what's the best move from the new position?
        if done:
            # Game over! No future rewards to consider
            future_value = 0
        else:
            # Choose future value based on learning strategy
            if self.strategy == 'sarsa':
                # 🎯 SARSA: Use the action actually taken from next_state (on-policy)
                # This makes SARSA more conservative - it learns the actual policy including exploration
                action_next, _ = self.choose_action(next_state, training=True)
                future_value = self.q_table[next_state, action_next]
            else:
                # 🎯 Q-LEARNING: Use the best action from next_state (off-policy)
                # This makes Q-learning more aggressive - it always learns the optimal path
                best_action_next = int(np.argmax(self.q_table[next_state]))
                future_value = self.q_table[next_state, best_action_next]
                
                # Check if the best action from next_state is the reverse of the action we just took
                # This prevents the agent from learning oscillatory behavior (going back and forth)
                # UP (0) <-> DOWN (1) are reverses, LEFT (2) <-> RIGHT (3) are reverses
                reverse_actions = {
                    0: 1,  # UP <-> DOWN
                    1: 0,  # DOWN <-> UP
                    2: 3,  # LEFT <-> RIGHT
                    3: 2,  # RIGHT <-> LEFT
                }
                
                # If the best action from next_state would take us back to where we came from,
                # ignore the future value to prevent oscillating
                if reverse_actions.get(action) == best_action_next:
                    future_value = 0
        
        # Step 3: The Q-learning formula!
        # This calculates how good this move REALLY was
        # It considers both the immediate reward AND future rewards
        new_value = old_value + self.lr * (reward + self.gamma * future_value - old_value)
        
        # Step 4: Save this new knowledge in our Q-table!
        self.q_table[state, action] = new_value
        
        # Step 5: Track total rewards (for monitoring progress)
        self.total_reward += reward
    
    def update_epsilon(self):
        """
        📉 Slowly reduce exploration over time.
        
        At first, we want to explore a lot to learn about the world.
        As we learn more, we can rely on our knowledge more.
        
        But we never stop exploring completely! (that's why we have epsilon_min)
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # Make sure we don't go below the minimum
            self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def get_best_action(self, state: int) -> int:
        """
        🏆 Get the best known move for a position.
        
        This is used when we're NOT training - just playing!
        We always pick the move with the highest Q-value.
        
        Args:
            state: Current position
            
        Returns:
            action: Best move (0-3)
        """
        return int(np.argmax(self.q_table[state]))
    
    def get_q_values(self, state: int) -> np.ndarray:
        """
        📊 Get all Q-values for a position.
        
        This shows how good each move is from this position.
        Useful for explaining what the robot is thinking!
        
        Args:
            state: Current position
            
        Returns:
            q_values: Array of 4 numbers [UP, DOWN, LEFT, RIGHT]
        """
        return self.q_table[state].copy()
    
    def save(self, filepath: str):
        """
        💾 Save the robot's brain to a file.
        
        This saves the Q-table so we can use it later!
        
        Args:
            filepath: Where to save the brain (e.g., 'robot_brain.json')
        """
        data = {
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.lr,
            'discount_factor': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'strategy': self.strategy,
            'q_table': self.q_table.tolist(),
            'total_reward': self.total_reward,
            'episodes_completed': self.episodes_completed
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"🧠 Robot brain saved to {filepath}!")
    
    @classmethod
    def load(cls, filepath: str) -> 'QLearningAgent':
        """
        📂 Load a robot brain from a file.
        
        This brings back a trained robot!
        
        Args:
            filepath: Where the brain is saved
            
        Returns:
            agent: A QLearningAgent with all the learned knowledge
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create a new agent with the saved settings (use default 'qlearning' if not in old saves)
        agent = cls(
            n_states=data['n_states'],
            n_actions=data['n_actions'],
            learning_rate=data['learning_rate'],
            discount_factor=data['discount_factor'],
            epsilon=data['epsilon'],
            epsilon_decay=data['epsilon_decay'],
            epsilon_min=data['epsilon_min'],
            strategy=data.get('strategy', 'qlearning')  # Default to qlearning for backward compatibility
        )
        
        # Restore the learned Q-table
        agent.q_table = np.array(data['q_table'])
        agent.total_reward = data.get('total_reward', 0)
        agent.episodes_completed = data.get('episodes_completed', 0)
        
        print(f"🧠 Robot brain loaded from {filepath}!")
        print(f"   Knowledge: {agent.n_states} positions × {agent.n_actions} actions")
        print(f"   Strategy: {agent.strategy.upper()}")
        
        return agent
    
    def print_q_table_summary(self):
        """
        📈 Print a summary of what the robot has learned.
        
        Shows statistics about the Q-table.
        """
        print("\n" + "=" * 50)
        print("🧠 ROBOT BRAIN SUMMARY")
        print("=" * 50)
        print(f"Strategy: {self.strategy.upper()}")
        print(f"Memory size: {self.n_states} positions × {self.n_actions} moves")
        print(f"Average Q-value: {np.mean(self.q_table):.2f}")
        print(f"Best Q-value: {np.max(self.q_table):.2f}")
        print(f"Episodes completed: {self.episodes_completed}")
        print(f"Total reward earned: {self.total_reward:.2f}")
        print(f"Current exploration rate (epsilon): {self.epsilon:.3f}")
        
        # Count how many moves have been learned (non-zero Q-values)
        learned_moves = np.count_nonzero(self.q_table)
        total_moves = self.n_states * self.n_actions
        learned_percent = (learned_moves / total_moves) * 100
        
        print(f"Moves learned: {learned_moves}/{total_moves} ({learned_percent:.1f}%)")
        print("=" * 50)


# Test the agent if we run this file directly
if __name__ == "__main__":
    print("🧠 Testing Q-Learning Robot!")
    print("=" * 50)
    
    # Create a small robot for a 5x5 grid (25 positions, 4 actions)
    agent = QLearningAgent(n_states=25, n_actions=4)
    
    print(f"\n🤖 Created a robot with {agent.n_states} positions and {agent.n_actions} moves")
    print(f"   Learning rate: {agent.lr}")
    print(f"   Discount factor: {agent.gamma}")
    print(f"   Exploration rate (epsilon): {agent.epsilon}")
    
    # Show empty Q-table
    print("\n📊 Q-table (all zeros at first):")
    print(agent.q_table[:5])  # Show first 5 rows
    
    # Simulate some learning
    print("\n🎓 Simulating learning...")
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        
        # Start at position 0
        state = 0
        
        for step in range(5):
            # Choose an action
            action, is_exploration = agent.choose_action(state)
            action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
            print(f"  Step {step + 1}: At position {state}, chose {action_names[action]}")
            
            # Simulate a reward (just for testing)
            reward = random.randint(-5, 10)
            next_state = random.randint(0, 24)
            done = step == 4
            
            # Learn from this experience
            agent.learn(state, action, reward, next_state, done)
            
            print(f"    Got reward: {reward}, moved to position {next_state}")
            
            state = next_state
            if done:
                break
        
        agent.episodes_completed += 1
        agent.update_epsilon()
    
    # Show learned Q-table
    print("\n📊 Q-table after learning (first 5 positions):")
    print(agent.q_table[:5])
    
    # Print summary
    agent.print_q_table_summary()
    
    # Test save and load
    print("\n💾 Testing save/load...")
    agent.save("test_brain.json")
    loaded_agent = QLearningAgent.load("test_brain.json")
    
    print("\n✅ Q-Learning Agent test complete!")
