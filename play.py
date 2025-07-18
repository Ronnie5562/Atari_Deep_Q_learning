import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import time
import os

def create_environment(render_mode="human"):
    """Create and configure the Atari environment for playing"""
    # Create Atari environment with rendering
    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=42)
    # Frame stacking: stack 4 frames for temporal information
    env = VecFrameStack(env, n_stack=4)
    return env

def play_dqn_agent(model_path="dqn_model.zip", num_episodes=5):
    """
    Load trained DQN model and play episodes

    Args:
        model_path (str): Path to the saved model
        num_episodes (int): Number of episodes to play
    """

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Make sure you've trained the model first by running train.py")
        return

    # Create environment
    env = create_environment()

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)
    print("Model loaded successfully!")

    # Play episodes
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1} ===")

        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Use the trained model to predict action (greedy policy)
            action, _ = model.predict(obs, deterministic=True)

            # Take action in environment
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]  # VecEnv returns array
            steps += 1

            # Add small delay to make it easier to watch
            time.sleep(0.01)

            # Check if episode is done
            if done[0]:
                break

        print(f"Episode {episode + 1} finished!")
        print(f"Total reward: {total_reward}")
        print(f"Total steps: {steps}")

        # Wait a bit between episodes
        time.sleep(1)

    env.close()
    print("\nPlayback completed!")

def evaluate_agent_performance(model_path="dqn_model.zip", num_episodes=10):
    """
    Evaluate the agent's performance over multiple episodes

    Args:
        model_path (str): Path to the saved model
        num_episodes (int): Number of episodes for evaluation
    """

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return

    # Create environment without rendering for faster evaluation
    env = create_environment()

    # Load the trained model
    print(f"Loading model for evaluation...")
    model = DQN.load(model_path)

    rewards = []
    episode_lengths = []

    print(f"Evaluating agent over {num_episodes} episodes...")

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Use greedy policy (deterministic=True)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]
            steps += 1

            if done[0]:
                break

        rewards.append(total_reward)
        episode_lengths.append(steps)

        if (episode + 1) % 5 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes")

    env.close()

    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(episode_lengths)

    print(f"\n=== Evaluation Results ===")
    print(f"Number of episodes: {num_episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.2f}")
    print(f"Best episode reward: {max(rewards):.2f}")
    print(f"Worst episode reward: {min(rewards):.2f}")

    return rewards, episode_lengths

def play_with_statistics(model_path="dqn_model.zip", num_episodes=3):
    """
    Play episodes while showing detailed statistics
    """

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return

    # Create environment
    env = create_environment()

    # Load the trained model
    model = DQN.load(model_path)

    all_rewards = []
    all_lengths = []

    for episode in range(num_episodes):
        print(f"\n{'='*20} Episode {episode + 1} {'='*20}")

        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False

        # Track actions for statistics
        actions_taken = []

        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            actions_taken.append(action[0])

            # Take action
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]
            steps += 1

            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"Step {steps}: Reward = {total_reward:.1f}")

            time.sleep(0.01)  # Small delay for visibility

            if done[0]:
                break

        all_rewards.append(total_reward)
        all_lengths.append(steps)

        # Episode statistics
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Episode length: {steps} steps")
        print(f"  Actions distribution: {np.bincount(actions_taken)}")

        time.sleep(2)  # Pause between episodes

    env.close()

    # Overall statistics
    print(f"\n{'='*20} Overall Statistics {'='*20}")
    print(f"Mean reward: {np.mean(all_rewards):.2f}")
    print(f"Mean episode length: {np.mean(all_lengths):.2f}")

def main():
    """Main function"""

    print("DQN Agent Player for BreakoutNoFrameskip-v4")
    print("=" * 50)

    # Check if model exists
    model_path = "dqn_model.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first by running 'python train.py'")
        return

    # Menu for different play modes
    print("\nChoose play mode:")
    print("1. Play with visualization (recommended)")
    print("2. Evaluate performance (no visualization)")
    print("3. Play with detailed statistics")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        print("\nStarting visual playback...")
        print("Close the game window to stop playing.")
        num_episodes = int(input("Number of episodes to play (default 3): ") or "3")
        play_dqn_agent(model_path, num_episodes)

    elif choice == "2":
        print("\nStarting performance evaluation...")
        num_episodes = int(input("Number of episodes for evaluation (default 10): ") or "10")
        evaluate_agent_performance(model_path, num_episodes)

    elif choice == "3":
        print("\nStarting detailed statistics playback...")
        num_episodes = int(input("Number of episodes to play (default 3): ") or "3")
        play_with_statistics(model_path, num_episodes)

    else:
        print("Invalid choice. Running default visual playback...")
        play_dqn_agent(model_path, 3)

if __name__ == "__main__":
    main()
