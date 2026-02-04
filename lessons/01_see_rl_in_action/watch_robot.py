"""
🎬 LESSON 1: SEE RL IN ACTION! 🎬

Welcome to your first mission, Space Cadet! 👨‍🚀

In this lesson, you'll watch a PRE-TRAINED robot navigate a space station.
No coding required - just watch and learn!

WHAT YOU'LL SEE:
- 🤖 A smart robot moving around
- 💥 Asteroids to avoid
- ⚡ Energy stars to collect
- ⭐ A goal to reach
- 🧠 The robot's Q-values (its memory!)

RUN THIS FILE:
    python lessons/01_see_rl_in_action/watch_robot.py

Then watch the magic happen! ✨
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.trainer import RobotTrainer
from colorama import init, Fore, Style

init(autoreset=True)


def main():
    """
    🚀 Main function for Lesson 1.
    """
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}🎬 LESSON 1: SEE REINFORCEMENT LEARNING IN ACTION!")
    print(f"{Fore.CYAN}{'='*60}\n")
    
    print(f"{Fore.WHITE}Welcome, Space Cadet! 👨‍🚀")
    print(f"\nIn this lesson, you'll watch a robot that has ALREADY learned")
    print(f"to navigate a space station. Watch carefully to understand:")
    print(f"\n  {Fore.GREEN}• STATE: Where is the robot?")
    print(f"  {Fore.BLUE}• ACTION: What move does it choose?")
    print(f"  {Fore.MAGENTA}• REWARD: Does it get points or lose points?")
    print(f"  {Fore.CYAN}• Q-VALUES: Why did it choose that move?\n")
    
    print(f"{Fore.YELLOW}🎮 PRESS ENTER TO START WATCHING!{Style.RESET_ALL}")
    input()
    
    # Create a trainer with a pre-trained robot
    # First, let's train a robot quickly
    print(f"\n{Fore.GREEN}🤖 Training a robot for you to watch...")
    print(f"{Fore.WHITE}This will take about 30 seconds...\n")
    
    trainer = RobotTrainer(
        grid_size=5,
        num_asteroids=3,
        num_energy=2,
        use_visualizer=True,
        explain_mode='smart',  # Will explain interesting moments
        fps=5,  # Slower so kids can follow
    )
    
    # Train the robot
    trainer.train(
        n_episodes=200,  # Quick training
        save_path='lessons/01_see_rl_in_action/demo_robot.json',
        eval_interval=50,
    )
    
    print(f"\n{Fore.GREEN}✅ Robot trained! Now let's watch it play!\n")
    print(f"{Fore.YELLOW}🎮 PRESS ENTER TO WATCH THE TRAINED ROBOT!{Style.RESET_ALL}")
    input()
    
    # Now watch the trained robot
    print(f"\n{Fore.CYAN}🎬 WATCHING TRAINED ROBOT...")
    print(f"{Fore.WHITE}Notice how it avoids asteroids and goes for the goal!\n")
    
    # Evaluate with visualization
    avg_reward, success_rate = trainer.evaluate(
        n_episodes=5,
        render=True,
        slow_mode=True,  # Slow down so kids can see each move
    )
    
    # Summary
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}🎉 LESSON 1 COMPLETE!")
    print(f"{Fore.GREEN}{'='*60}\n")
    
    print(f"{Fore.WHITE}Great job watching! You just saw:")
    print(f"\n  {Fore.GREEN}✓ A robot learning to navigate")
    print(f"  {Fore.GREEN}✓ The robot avoiding dangers")
    print(f"  {Fore.GREEN}✓ The robot collecting rewards")
    print(f"  {Fore.GREEN}✓ The robot reaching the goal!")
    
    print(f"\n{Fore.CYAN}WHAT YOU LEARNED:")
    print(f"{Fore.WHITE}  • Reinforcement Learning is like learning to ride a bike")
    print(f"  • The robot tries things, gets feedback (rewards), and improves")
    print(f"  • Over time, the robot remembers the best moves")
    
    print(f"\n{Fore.YELLOW}🚀 READY FOR THE NEXT LESSON?")
    print(f"{Fore.WHITE}  In Lesson 2, YOU will train your own robot!")
    print(f"  Go to: lessons/02_space_adventure/")
    
    # Clean up
    trainer.close()


if __name__ == "__main__":
    main()
