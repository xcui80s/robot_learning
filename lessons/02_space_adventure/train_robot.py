"""
🚀 LESSON 2: SPACE ADVENTURE - TRAIN YOUR OWN ROBOT! 🚀

Congratulations, Space Cadet! You've completed Lesson 1 and now
you're ready to train your OWN robot! 👨‍🚀

IN THIS LESSON YOU WILL:
- Train a robot from scratch (it starts knowing NOTHING!)
- Watch it learn and improve over time
- Test your trained robot
- Tweak the rewards to see what happens!

HOW TO USE THIS FILE:
1. Just run it to train your first robot:
   python lessons/02_space_adventure/train_robot.py

2. Your trained robot will be saved as 'robot_brain.json'

3. Then test it with:
   python lessons/02_space_adventure/play_robot.py

EXPERIMENTS TO TRY:
After training once, try these changes in the code below:

🔧 TWEAK #1: Change the rewards
   Look for REWARD_GOAL, REWARD_ENERGY, REWARD_ASTEROID in the code
   Try making the asteroid penalty -50 instead of -10!
   What happens? The robot becomes super careful!

🔧 TWEAK #2: Change the grid size
   Try grid_size=8 for a bigger challenge!
   The robot will need more episodes to learn.

🔧 TWEAK #3: Change learning speed
   Try learning_rate=0.5 for faster learning!
   But watch out - it might become unstable.

🔧 TWEAK #4: More obstacles
   Try num_asteroids=8 for a hard mode!
   Can the robot still find a path?

Have fun experimenting! 🎉
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.trainer import RobotTrainer
from colorama import init, Fore, Style

init(autoreset=True)


def main():
    """
    🚀 Main function for Lesson 2 - Training your own robot!
    """
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}🚀 LESSON 2: TRAIN YOUR OWN ROBOT!")
    print(f"{Fore.CYAN}{'='*60}\n")
    
    print(f"{Fore.WHITE}Welcome back, Space Cadet! 👨‍🚀")
    print(f"\nIn this lesson, YOU will train a robot from scratch!")
    print(f"The robot starts knowing NOTHING and learns by trying.")
    
    print(f"\n{Fore.GREEN}What you'll do:")
    print(f"  1. Watch the robot explore randomly at first")
    print(f"  2. See it learn from rewards and mistakes")
    print(f"  3. Watch it get better over many episodes")
    print(f"  4. Test the trained robot!")
    
    print(f"\n{Fore.YELLOW}🎮 PRESS ENTER TO START TRAINING!{Style.RESET_ALL}")
    input()
    
    # =========================================================================
    # 🔧 CONFIGURATION - You can change these values!
    # =========================================================================
    
    # Grid settings
    GRID_SIZE = 5          # Try 8 for a bigger challenge!
    NUM_ASTEROIDS = 3      # Try 8 for hard mode!
    NUM_ENERGY = 2         # More energy = more points available
    
    # Learning settings
    LEARNING_RATE = 0.1    # How fast it learns (0.1 = slow and steady)
    N_EPISODES = 300       # How many training episodes (300 = good start)
    
    print(f"\n{Fore.CYAN}⚙️  Training Configuration:{Style.RESET_ALL}")
    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Asteroids: {NUM_ASTEROIDS}")
    print(f"  Energy stars: {NUM_ENERGY}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Episodes: {N_EPISODES}")
    
    print(f"\n{Fore.GREEN}🤖 Creating your robot trainer...")
    
    # Create the trainer with your settings
    trainer = RobotTrainer(
        grid_size=GRID_SIZE,
        num_asteroids=NUM_ASTEROIDS,
        num_energy=NUM_ENERGY,
        learning_rate=LEARNING_RATE,
        use_visualizer=True,      # Show the window!
        explain_mode='smart',     # Explain interesting moments
        fps=10,                   # Animation speed
    )
    
    print(f"{Fore.GREEN}✅ Trainer ready!\n")
    print(f"{Fore.YELLOW}🚀 Starting training in 3 seconds...")
    print(f"{Fore.WHITE}Watch the robot learn! Press Ctrl+C to stop early.\n")
    
    import time
    time.sleep(3)
    
    # =========================================================================
    # 🚀 TRAIN THE ROBOT!
    # =========================================================================
    
    print(f"{Fore.GREEN}🏋️  Training in progress...\n")
    
    # This is where the magic happens!
    # The robot will play N_EPISODES games and learn from each one
    rewards, lengths = trainer.train(
        n_episodes=N_EPISODES,
        save_path='lessons/02_space_adventure/robot_brain.json',
        eval_interval=50,  # Show progress every 50 episodes
    )
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}🎉 TRAINING COMPLETE!")
    print(f"{Fore.GREEN}{'='*60}\n")
    
    print(f"{Fore.WHITE}Your robot has been trained and saved!")
    print(f"  Location: lessons/02_space_adventure/robot_brain.json")
    print(f"\n{Fore.CYAN}📊 Training Results:{Style.RESET_ALL}")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Best score: {max(rewards):.1f} points")
    print(f"  Final average: {sum(rewards[-100:])/len(rewards[-100:]):.1f} points")
    
    # =========================================================================
    # 🎮 TEST THE TRAINED ROBOT
    # =========================================================================
    
    print(f"\n{Fore.YELLOW}🎮 Now let's test your trained robot!")
    print(f"{Fore.WHITE}Watch it use what it learned (no more random moves)\n")
    print(f"{Fore.YELLOW}PRESS ENTER TO TEST!{Style.RESET_ALL}")
    input()
    
    avg_reward, success_rate = trainer.evaluate(
        n_episodes=5,
        render=True,      # Show the window
        slow_mode=True,   # Slow down so you can watch
    )
    
    # =========================================================================
    # 🎓 WRAP UP
    # =========================================================================
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}🎓 LESSON 2 COMPLETE!")
    print(f"{Fore.GREEN}{'='*60}\n")
    
    print(f"{Fore.WHITE}Congratulations! You just trained a robot using Q-Learning! 🎉")
    
    print(f"\n{Fore.CYAN}WHAT YOU ACCOMPLISHED:{Style.RESET_ALL}")
    print(f"  ✅ Created a reinforcement learning agent")
    print(f"  ✅ Trained it through {N_EPISODES} episodes")
    print(f"  ✅ Watched it learn from trial and error")
    print(f"  ✅ Tested the trained robot")
    
    print(f"\n{Fore.CYAN}WHAT YOU LEARNED:{Style.RESET_ALL}")
    print(f"  • The robot starts knowing nothing (random moves)")
    print(f"  • It learns by getting rewards and penalties")
    print(f"  • Over time, it remembers the best moves (Q-values)")
    print(f"  • Eventually, it becomes an expert!")
    
    print(f"\n{Fore.YELLOW}🔬 EXPERIMENTS TO TRY:{Style.RESET_ALL}")
    print(f"  1. Open this file and change GRID_SIZE to 8")
    print(f"  2. Change NUM_ASTEROIDS to 8 for a challenge")
    print(f"  3. Change REWARD_ASTEROID to -50 (super careful robot)")
    print(f"  4. Try LEARNING_RATE = 0.5 (faster but less stable)")
    
    print(f"\n{Fore.GREEN}🚀 READY FOR MORE?{Style.RESET_ALL}")
    print(f"  Run 'python play_robot.py' to watch your robot anytime!")
    print(f"  Each time you train, it creates a new robot_brain.json")
    
    # Clean up
    trainer.close()
    
    print(f"\n{Fore.CYAN}Thanks for learning! Keep experimenting! 🚀{Style.RESET_ALL}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Training stopped early. That's okay!")
        print(f"Your robot was still learning. Try again with fewer episodes.")
