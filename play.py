import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
import os
import warnings
from collections import deque

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ManualFrameStacker:
    """Manual frame stacking implementation with proper channel ordering"""
    def __init__(self, env, num_stack=4):
        self.env = env
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        # Update observation space to match model expectations
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(num_stack, 84, 84),
            dtype=np.uint8
        )
    
    def reset(self):
        obs, info = self.env.reset()
        # Squeeze channel dimension and store
        obs = obs.squeeze(-1)
        # Initialize with the first frame repeated
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return np.stack(self.frames), info
    
    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        # Squeeze channel dimension
        next_obs = next_obs.squeeze(-1)
        self.frames.append(next_obs)
        stacked_obs = np.stack(self.frames)
        return stacked_obs, reward, terminated, truncated, info
    
    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()

def create_environment(render_mode=None):
    """Create and configure the Atari environment for playing"""
    # Create base environment
    env = gym.make('ALE/Breakout-v5', 
                   render_mode=render_mode,
                   full_action_space=False,
                   repeat_action_probability=0.0)
    
    # Apply preprocessing using Stable Baselines3 wrapper
    env = AtariWrapper(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        clip_reward=False
    )
    
    # Apply manual frame stacking
    env = ManualFrameStacker(env, num_stack=4)
    return env

def run_agent(num_episodes, model_path="dqn_model.zip", mode='visual'):
    """
    Unified function to run the agent with different visualization modes
    
    Args:
        num_episodes (int): Number of episodes to run
        model_path (str): Path to the trained model file
        mode (str): Execution mode ('visual', 'evaluate', or 'stats')
    """
    # Validate inputs
    if mode not in ['visual', 'evaluate', 'stats']:
        print(f"Invalid mode '{mode}'. Using 'visual' mode.")
        mode = 'visual'
    
    if num_episodes < 1:
        print(f"Invalid episode count {num_episodes}. Using 3 episodes.")
        num_episodes = 3
    
    # Check model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Make sure you've trained the model first.")
        return
    
    # Create environment based on mode
    render_mode = "human" if mode in ['visual', 'stats'] else None
    env = create_environment(render_mode=render_mode)
    
    # Load the trained model
    model = DQN.load(model_path)
    
    rewards = []
    episode_lengths = []
    all_actions = []
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        actions_taken = []
        
        while not (terminated or truncated):
            # Prepare observation for model
            obs_batch = np.expand_dims(obs, axis=0)
            
            # Predict action
            action, _ = model.predict(obs_batch, deterministic=True)
            action_val = action[0]
            actions_taken.append(action_val)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action_val)
            total_reward += reward
            steps += 1
            
            # Add delay for visual modes
            if mode in ['visual', 'stats']:
                time.sleep(0.01)
        
        # Record results
        rewards.append(total_reward)
        episode_lengths.append(steps)
        all_actions.append(actions_taken)
        
        # Print episode summary
        if mode in ['visual', 'stats']:
            print(f"\nEpisode {episode + 1}/{num_episodes} completed")
            print(f"  Reward: {total_reward:.1f}")
            print(f"  Steps: {steps}")
            
            if mode == 'stats':
                action_counts = np.bincount(actions_taken)
                print("  Actions taken:")
                for action_idx, count in enumerate(action_counts):
                    print(f"    Action {action_idx}: {count} times")
    
    # Calculate performance metrics
    total_time = time.time() - start_time
    mean_reward = np.mean(rewards)
    mean_length = np.mean(episode_lengths)
    
    # Print final summary
    print(f"\n=== Summary ===")
    print(f"Total episodes: {num_episodes}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average steps per second: {np.sum(episode_lengths)/total_time:.1f}")
    print(f"Average reward: {mean_reward:.1f}")
    print(f"Average steps per episode: {mean_length:.1f}")
    
    if mode != 'visual':
        print(f"Best reward: {max(rewards):.1f}")
        print(f"Worst reward: {min(rewards):.1f}")
        print(f"Reward standard deviation: {np.std(rewards):.1f}")
    
    # Additional detailed statistics for stats mode
    if mode == 'stats':
        # Combine all actions across episodes
        all_actions_flat = [action for episode_actions in all_actions for action in episode_actions]
        total_actions = len(all_actions_flat)
        
        print("\n=== Detailed Action Statistics ===")
        print(f"Total actions taken: {total_actions}")
        
        # Calculate action frequencies
        action_counts = np.bincount(all_actions_flat)
        for action_idx, count in enumerate(action_counts):
            percentage = (count / total_actions) * 100
            print(f"Action {action_idx}: {count} times ({percentage:.1f}%)")
        
        # Calculate actions per episode
        print("\nActions per episode:")
        for episode in range(num_episodes):
            print(f"Episode {episode+1}: {len(all_actions[episode])} actions")
    
    env.close()


if __name__ == "__main__":
    # Clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("DQN Agent Player for ALE/Breakout-v5")
    print("=" * 50)
    
    # Prompt for number of episodes
    while True:
        try:
            num_episodes = int(input("\nEnter the number of episodes to play: "))
            if num_episodes > 0:
                break
            else:
                print("Please enter a number greater than 0.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Prompt for mode selection
    print("\nSelect an execution mode:")
    print("1. Visual Playback (with rendering)")
    print("2. Performance Evaluation (no rendering)")
    print("3. Detailed Action Statistics (with rendering and action analysis)")
    
    mode_choice = input("Enter your choice (1-3): ")
    mode_map = {
        '1': 'visual',
        '2': 'evaluate',
        '3': 'stats'
    }
    
    mode = mode_map.get(mode_choice, 'visual')
    
    # Run the agent with selected options
    run_agent(num_episodes=num_episodes, mode=mode)
    
    # Keep terminal open
    input("\nPress Enter to exit...")
