"""
📚 CONCEPT EXPLAINER 📚

This module teaches kids the basic concepts of Reinforcement Learning
in real-time while the robot is learning!

It explains:
- 🤖 STATE: Where is the robot? What does it see?
- 🎮 ACTION: What move did the robot choose?
- ⭐ REWARD: Did the robot get points or lose points?
- 🧠 Q-VALUES: Why did the robot choose that move?

The explanations are kid-friendly and happen during training,
so kids can watch and learn at the same time!
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from colorama import init, Fore, Back, Style

# Initialize colorama for colored terminal output
init(autoreset=True)


class ConceptExplainer:
    """
    🎓 Explains RL concepts in a way kids can understand!
    
    This class provides:
    - Real-time explanations during training
    - Visual diagrams and examples
    - Interactive "try this" prompts
    - Simple language for ages 10-14
    
    MODES:
    - 'verbose': Explain everything on every step
    - 'smart': Only explain when something interesting happens
    - 'minimal': Just show key learning moments
    """
    
    def __init__(self, mode: str = 'smart', grid_size: int = 5):
        """
        🎓 Set up the concept explainer.
        
        Args:
            mode: How much explanation to show ('verbose', 'smart', 'minimal')
            grid_size: Size of the grid for coordinate explanations
        """
        self.mode = mode
        self.grid_size = grid_size
        self.step_count = 0
        self.episode_count = 0
        self.last_explanation = ""
        
        # Track interesting events
        self.last_reward = 0
        self.events_history = []
        
    def explain_step(
        self,
        env_info: Dict,
        state: int,
        action: int,
        reward: float,
        q_values: Optional[np.ndarray] = None,
        training_mode: bool = True
    ) -> str:
        """
        🎬 Explain what just happened in the robot's learning!
        
        This is the main function that explains each step of RL.
        
        Args:
            env_info: Environment information (positions, objects)
            state: Current state number
            action: Action taken (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            reward: Reward received
            q_values: Q-values for current state (optional)
            training_mode: Is the robot still learning?
            
        Returns:
            explanation: String with the explanation (also prints to console)
        """
        self.step_count += 1
        
        # Determine if we should explain this step
        should_explain = self._should_explain(reward, state)
        
        if not should_explain and self.mode != 'verbose':
            return ""
        
        # Build the explanation
        explanation = []
        
        # Header
        explanation.append(self._format_header())
        
        # Explain STATE
        explanation.append(self._explain_state(env_info, state))
        
        # Explain ACTION
        explanation.append(self._explain_action(action, q_values))
        
        # Explain REWARD
        explanation.append(self._explain_reward(reward, env_info))
        
        # Explain Q-values (if available)
        if q_values is not None and training_mode:
            explanation.append(self._explain_q_values(q_values, action))
        
        # Learning tip
        if training_mode and self.step_count % 5 == 0:
            explanation.append(self._give_learning_tip())
        
        # Footer
        explanation.append(self._format_footer())
        
        # Combine and print
        full_explanation = "\n".join(explanation)
        print(full_explanation)
        
        self.last_explanation = full_explanation
        self.last_reward = reward
        
        return full_explanation
    
    def _should_explain(self, reward: float, state: int) -> bool:
        """
        🤔 Decide if this step is worth explaining.
        
        In 'smart' mode, we only explain interesting events:
        - First step of an episode
        - Got a big reward (positive or negative)
        - Reached a new state we haven't seen
        - Every 10 steps to show progress
        """
        if self.mode == 'verbose':
            return True
        
        if self.mode == 'minimal':
            # Only explain big events
            return abs(reward) >= 10 or self.step_count == 1
        
        # Smart mode: explain interesting things
        interesting = (
            self.step_count == 1 or  # First step
            abs(reward) >= 5 or  # Big reward (positive or negative)
            abs(reward - self.last_reward) > 5 or  # Big change in reward
            self.step_count % 10 == 0  # Every 10 steps
        )
        
        return interesting
    
    def _format_header(self) -> str:
        """
        🎨 Format the explanation header.
        """
        width = 60
        header = f"\n{Fore.CYAN}{'=' * width}"
        header += f"\n{Fore.YELLOW}🤖 ROBOT LEARNING - STEP {self.step_count}"
        header += f"\n{Fore.CYAN}{'=' * width}"
        return header
    
    def _format_footer(self) -> str:
        """
        🎨 Format the explanation footer.
        """
        width = 60
        return f"{Fore.CYAN}{'=' * width}\n"
    
    def _explain_state(self, env_info: Dict, state: int) -> str:
        """
        📍 Explain the current STATE.
        
        Shows where the robot is and what it can see.
        """
        robot_pos = env_info.get('robot_pos', (0, 0))
        row, col = robot_pos
        
        asteroids = env_info.get('asteroids', [])
        energy_stars = env_info.get('energy_stars', [])
        goal_pos = env_info.get('goal_pos', (4, 4))
        
        explanation = f"\n{Fore.GREEN}📍 STATE: Where is the robot?\n"
        explanation += f"{Fore.WHITE}   Position: ({row}, {col}) [State #{state}]\n"
        
        # What can the robot see nearby?
        nearby = []
        
        # Check adjacent cells
        for dr, dc, direction in [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]:
            check_pos = (row + dr, col + dc)
            if 0 <= check_pos[0] < self.grid_size and 0 <= check_pos[1] < self.grid_size:
                if check_pos in asteroids:
                    nearby.append(f"💥 Asteroid {direction}")
                elif check_pos in energy_stars:
                    nearby.append(f"⚡ Energy star {direction}")
                elif check_pos == goal_pos:
                    nearby.append(f"⭐ GOAL {direction}!")
        
        if nearby:
            explanation += f"\n{Fore.YELLOW}   What the robot sees:\n"
            for item in nearby:
                explanation += f"{Fore.WHITE}     • {item}\n"
        else:
            explanation += f"{Fore.WHITE}   The robot sees empty space around it.\n"
        
        explanation += f"\n{Fore.CYAN}   💡 STATE is like a photo showing exactly where the robot is\n"
        explanation += f"      and what's nearby. The robot needs to know this to make good choices!\n"
        
        return explanation
    
    def _explain_action(self, action: int, q_values: Optional[np.ndarray] = None) -> str:
        """
        🎮 Explain the chosen ACTION.
        
        Shows what move the robot made and why.
        """
        action_names = ["UP ⬆️", "DOWN ⬇️", "LEFT ⬅️", "RIGHT ➡️"]
        action_name = action_names[action]
        
        explanation = f"\n{Fore.BLUE}🎮 ACTION: What did the robot do?\n"
        explanation += f"{Fore.WHITE}   Robot chose: {action_name}\n"
        
        if q_values is not None:
            # Explain why this action was chosen
            best_q = np.max(q_values)
            current_q = q_values[action]
            
            if current_q == best_q:
                explanation += f"{Fore.GREEN}   This was the BEST move the robot knew about!\n"
                explanation += f"{Fore.WHITE}   Q-value: {current_q:.1f} (the highest)\n"
            else:
                diff = best_q - current_q
                explanation += f"{Fore.YELLOW}   This wasn't the best move (exploring!)\n"
                explanation += f"{Fore.WHITE}   Q-value: {current_q:.1f} (best is {best_q:.1f})\n"
                explanation += f"{Fore.CYAN}   The robot is trying new things to learn!\n"
        
        explanation += f"\n{Fore.CYAN}   💡 ACTION is like a choice: UP, DOWN, LEFT, or RIGHT\n"
        explanation += f"      The robot picks one based on what it has learned so far.\n"
        
        return explanation
    
    def _explain_reward(self, reward: float, env_info: Dict) -> str:
        """
        ⭐ Explain the REWARD.
        
        Shows if the robot got points or lost points, and why.
        """
        explanation = f"\n{Fore.MAGENTA}⭐ REWARD: Did the robot do good or bad?\n"
        
        # Determine what happened based on reward
        if reward >= 100:
            explanation += f"{Fore.GREEN}   🎉 REACHED THE GOAL! +{reward:.0f} points!\n"
            explanation += f"{Fore.WHITE}   The robot successfully navigated to the star!\n"
        elif reward >= 10:
            explanation += f"{Fore.GREEN}   ⭐ COLLECTED ENERGY! +{reward:.0f} points!\n"
            explanation += f"{Fore.WHITE}   The robot found an energy star! Great job!\n"
        elif reward <= -10:
            explanation += f"{Fore.RED}   💥 HIT ASTEROID! {reward:.0f} points\n"
            explanation += f"{Fore.WHITE}   Oh no! The robot crashed into an asteroid!\n"
            explanation += f"{Fore.YELLOW}   The robot will remember: 'Don't go that way!'\n"
        elif reward == -1:
            explanation += f"{Fore.YELLOW}   ⏱️  Small step penalty: {reward:.0f} points\n"
            explanation += f"{Fore.WHITE}   The robot used one move. This encourages finding\n"
            explanation += f"{Fore.WHITE}   the shortest path!\n"
        else:
            explanation += f"{Fore.WHITE}   Reward: {reward:+.1f} points\n"
        
        explanation += f"\n{Fore.CYAN}   💡 REWARD is like a score in a video game:\n"
        explanation += f"      • Good things = +points (green)\n"
        explanation += f"      • Bad things = -points (red)\n"
        explanation += f"      • The robot tries to get MORE points!\n"
        
        return explanation
    
    def _explain_q_values(self, q_values: np.ndarray, chosen_action: int) -> str:
        """
        🧠 Explain the Q-values (robot's memory).
        
        Shows what the robot knows about each possible move.
        """
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        explanation = f"\n{Fore.CYAN}🧠 ROBOT'S MEMORY (Q-values):\n"
        explanation += f"{Fore.WHITE}   These numbers show how good each move is from here:\n\n"
        
        # Show each action with its Q-value
        for i, (name, q_val) in enumerate(zip(action_names, q_values)):
            if i == chosen_action:
                marker = f"{Fore.GREEN}★ "  # Star for chosen action
            else:
                marker = f"{Fore.WHITE}  "
            
            # Color-code the value
            if q_val > 50:
                color = Fore.GREEN
                emoji = "🟢"
            elif q_val > 0:
                color = Fore.YELLOW
                emoji = "🟡"
            elif q_val > -20:
                color = Fore.WHITE
                emoji = "⚪"
            else:
                color = Fore.RED
                emoji = "🔴"
            
            explanation += f"{marker}{color}{emoji} {name}: {q_val:+6.1f}\n"
        
        explanation += f"\n{Fore.CYAN}   💡 Q-values are like the robot's memory:\n"
        explanation += f"      • Higher numbers = better moves\n"
        explanation += f"      • Lower numbers = worse moves\n"
        explanation += f"      • The robot updates these after every move!\n"
        
        return explanation
    
    def _give_learning_tip(self) -> str:
        """
        💡 Give a random learning tip.
        """
        tips = [
            f"{Fore.YELLOW}\n💡 TIP: The robot is exploring! When epsilon is high,\n"
            f"   it tries random moves to discover new strategies.",
            
            f"{Fore.YELLOW}\n💡 TIP: Watch the Q-values change! They show what\n"
            f"   the robot is learning about each position.",
            
            f"{Fore.YELLOW}\n💡 TIP: State + Action = New State + Reward\n"
            f"   This is the learning loop! The robot repeats this many times.",
            
            f"{Fore.YELLOW}\n💡 TIP: The goal is worth 100 points!\n"
            f"   That's why the robot tries so hard to reach it!",
            
            f"{Fore.YELLOW}\n💡 TIP: Hitting asteroids gives negative points.\n"
            f"   The robot learns to avoid them!",
            
            f"{Fore.YELLOW}\n💡 TIP: The robot balances EXPLORATION (trying new things)\n"
            f"   and EXPLOITATION (using what it knows).",
        ]
        
        import random
        return random.choice(tips)
    
    def explain_episode_start(self, episode: int, epsilon: float):
        """
        🎬 Explain the start of a new episode.
        """
        self.episode_count = episode
        self.step_count = 0
        
        width = 60
        print(f"\n{Fore.CYAN}{'=' * width}")
        print(f"{Fore.GREEN}🎬 STARTING EPISODE #{episode}")
        print(f"{Fore.WHITE}   Exploration rate (epsilon): {epsilon:.3f}")
        
        if epsilon > 0.5:
            print(f"{Fore.YELLOW}   The robot is exploring a lot - trying new strategies!")
        else:
            print(f"{Fore.GREEN}   The robot is using what it learned - getting smarter!")
        
        print(f"{Fore.CYAN}{'=' * width}\n")
    
    def explain_episode_end(
        self, 
        episode: int, 
        total_reward: float, 
        steps: int, 
        success: bool,
        epsilon: float
    ):
        """
        🏁 Explain the end of an episode.
        """
        width = 60
        print(f"\n{Fore.CYAN}{'=' * width}")
        print(f"{Fore.GREEN}🏁 EPISODE #{episode} COMPLETE!")
        
        if success:
            print(f"{Fore.GREEN}   ✓ SUCCESS! Robot reached the goal!")
            print(f"{Fore.GREEN}   🎉 Total reward: {total_reward:.1f} points!")
        else:
            print(f"{Fore.RED}   ✗ Didn't reach goal this time")
            print(f"{Fore.YELLOW}   📊 Total reward: {total_reward:.1f} points")
        
        print(f"{Fore.WHITE}   Steps taken: {steps}")
        print(f"{Fore.WHITE}   New epsilon: {epsilon:.3f}")
        print(f"{Fore.CYAN}{'=' * width}\n")
    
    def explain_training_start(self, n_episodes: int):
        """
        🚀 Explain the start of training.
        """
        width = 60
        print(f"\n{Fore.GREEN}{'=' * width}")
        print(f"{Fore.YELLOW}🚀 TRAINING THE ROBOT!")
        print(f"{Fore.WHITE}   The robot will play {n_episodes} episodes")
        print(f"{Fore.WHITE}   Each episode, it learns a little more!")
        print(f"{Fore.CYAN}   Watch the Q-values and rewards change!")
        print(f"{Fore.GREEN}{'=' * width}\n")
    
    def explain_training_complete(self, total_episodes: int, best_score: float):
        """
        🎉 Explain that training is complete.
        """
        width = 60
        print(f"\n{Fore.GREEN}{'=' * width}")
        print(f"{Fore.YELLOW}🎉 TRAINING COMPLETE!")
        print(f"{Fore.WHITE}   Total episodes: {total_episodes}")
        print(f"{Fore.GREEN}   Best score: {best_score:.1f} points")
        print(f"{Fore.CYAN}   The robot has learned! Now let's test it!")
        print(f"{Fore.GREEN}{'=' * width}\n")
    
    def set_mode(self, mode: str):
        """
        🔄 Change explanation mode.
        
        Args:
            mode: 'verbose', 'smart', or 'minimal'
        """
        if mode in ['verbose', 'smart', 'minimal']:
            self.mode = mode
            print(f"{Fore.CYAN}Explanation mode set to: {mode}")
        else:
            print(f"{Fore.RED}Invalid mode. Choose: verbose, smart, or minimal")


# Test the explainer if we run this file directly
if __name__ == "__main__":
    print("📚 Testing Concept Explainer!")
    print("=" * 50)
    
    # Create explainer
    explainer = ConceptExplainer(mode='verbose', grid_size=5)
    
    # Simulate training steps
    explainer.explain_training_start(n_episodes=5)
    
    for episode in range(1, 3):
        epsilon = 1.0 - (episode - 1) * 0.2
        explainer.explain_episode_start(episode, epsilon)
        
        # Simulate some steps
        for step in range(3):
            env_info = {
                'robot_pos': (step, step),
                'asteroids': [(1, 1)],
                'energy_stars': [(2, 2)] if step < 2 else [],
                'goal_pos': (4, 4),
                'collected_energy': step,
                'steps': step,
            }
            
            state = step * 5 + step
            action = step % 4
            
            # Simulate rewards
            if step == 0:
                reward = -1
            elif step == 1:
                reward = 10  # Energy!
            else:
                reward = -10  # Ouch, asteroid!
            
            q_values = np.array([5.0, 2.0, 8.0, 3.0]) + np.random.randn(4) * 2
            
            explainer.explain_step(env_info, state, action, reward, q_values)
        
        explainer.explain_episode_end(
            episode=episode,
            total_reward=15.0,
            steps=3,
            success=False,
            epsilon=epsilon * 0.95
        )
    
    explainer.explain_training_complete(total_episodes=5, best_score=100.0)
    
    print("\n✅ Concept Explainer test complete!")
