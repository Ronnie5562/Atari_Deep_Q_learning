import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
import os
import time


def create_environment():
    """Create and configure the Atari environment"""
    # Create Atari environment with proper preprocessing
    env = make_atari_env('ALE/Breakout-v5', n_envs=1, seed=42)
    # Frame stacking: stack 4 frames for temporal information
    env = VecFrameStack(env, n_stack=4)
    return env


def train_dqn_agent(hyperparams, policy_type="CnnPolicy", total_timesteps=1000000):
    """
    Train a DQN agent with specified hyperparameters

    Args:
        hyperparams (dict): Dictionary containing hyperparameters
        policy_type (str): Either "CnnPolicy" or "MlpPolicy"
        total_timesteps (int): Total training timesteps
    """

    # Create environment
    env = create_environment()

    # Set up logging
    log_dir = f"./logs/dqn_breakout_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)

    # Configure logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # Create DQN model with hyperparameters
    model = DQN(
        policy_type,
        env,
        learning_rate=hyperparams['learning_rate'],
        gamma=hyperparams['gamma'],
        batch_size=hyperparams['batch_size'],
        exploration_initial_eps=hyperparams['epsilon_start'],
        exploration_final_eps=hyperparams['epsilon_end'],
        exploration_fraction=hyperparams['epsilon_decay'],
        buffer_size=100000,  # Replay buffer size
        learning_starts=50000,  # Start learning after this many steps
        target_update_interval=10000,  # Update target network every N steps
        train_freq=4,  # Train every 4 steps
        verbose=1,
        tensorboard_log=log_dir,
        device='auto'  # Use GPU if available
    )

    # Set the logger
    model.set_logger(new_logger)

    # Create evaluation environment
    eval_env = create_environment()

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print(f"Training DQN agent with {policy_type} policy...")
    print(f"Hyperparameters: {hyperparams}")
    print(f"Total timesteps: {total_timesteps}")

    # Train the model
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    training_time = time.time() - start_time

    # Save the final model
    model_path = "dqn_model.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print(f"Training completed in {training_time:.2f} seconds")

    # Close environments
    env.close()
    eval_env.close()

    return model, log_dir


def compare_policies():
    """Compare CNN and MLP policies"""

    # Default hyperparameters for comparison
    default_hyperparams = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.1
    }

    print("=== Comparing CNN vs MLP Policy ===")

    # Train with CNN Policy (recommended for Atari)
    print("\n1. Training with CNN Policy...")
    cnn_model, cnn_log_dir = train_dqn_agent(
        default_hyperparams,
        policy_type="CnnPolicy",
        total_timesteps=100000  # Reduced for quick comparison
    )

    # Note: MLP Policy is not recommended for Atari games as it doesn't handle
    # image inputs well, but we can try it for comparison
    print("\n2. Training with MLP Policy...")
    try:
        mlp_model, mlp_log_dir = train_dqn_agent(
            default_hyperparams,
            policy_type="MlpPolicy",
            total_timesteps=100000
        )
    except Exception as e:
        print(f"MLP Policy failed (expected for Atari): {e}")
        print("CNN Policy is the correct choice for Atari environments.")


def hyperparameter_tuning():
    """
    Perform hyperparameter tuning experiments
    """

    # Define different hyperparameter configurations to test
    hyperparameter_sets = [
        {
            'name': 'Set 1 - Default',
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        {
            'name': 'Set 2 - Higher Learning Rate',
            'learning_rate': 5e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        {
            'name': 'Set 3 - Lower Gamma',
            'learning_rate': 1e-4,
            'gamma': 0.95,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        {
            'name': 'Set 4 - Larger Batch Size',
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 64,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        },
        {
            'name': 'Set 5 - Slower Epsilon Decay',
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.2
        }
    ]

    print("=== Hyperparameter Tuning Experiments ===")

    results = []

    for i, params in enumerate(hyperparameter_sets):
        print(f"\nExperiment {i+1}: {params['name']}")
        print(f"Parameters: {params}")

        # Remove 'name' from params for training
        training_params = {k: v for k, v in params.items() if k != 'name'}

        # Train with reduced timesteps for hyperparameter tuning
        model, log_dir = train_dqn_agent(
            training_params,
            policy_type="CnnPolicy",
            total_timesteps=200000  # Reduced for faster experimentation
        )

        results.append({
            'name': params['name'],
            'params': training_params,
            'log_dir': log_dir
        })

    # Print summary
    print("\n=== Hyperparameter Tuning Summary ===")
    for result in results:
        print(f"{result['name']}: {result['params']}")
        print(f"  Log directory: {result['log_dir']}")

    return results


def main():
    """Main function to run training"""

    print("DQN Training for ALE/Breakout-v5")
    print("=" * 50)

    # Choose what to run:
    # 1. Quick policy comparison
    # 2. Full hyperparameter tuning
    # 3. Single training run

    choice = input(
        "Choose option:\n1. Policy comparison\n2. Hyperparameter tuning\n3. Single training run\nEnter choice (1-3): ")

    if choice == "1":
        compare_policies()
    elif choice == "2":
        hyperparameter_tuning()
    elif choice == "3":
        # Single training run with default parameters
        default_hyperparams = {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        }

        model, log_dir = train_dqn_agent(
            default_hyperparams,
            policy_type="CnnPolicy",
            total_timesteps=1000000  # Full training
        )

        print(f"\nTraining completed!")
        print(f"Model saved as: dqn_model.zip")
        print(f"Logs saved in: {log_dir}")
    else:
        print("Invalid choice. Running single training run...")
        # Default to single training run
        default_hyperparams = {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.1
        }

        model, log_dir = train_dqn_agent(
            default_hyperparams,
            policy_type="CnnPolicy",
            total_timesteps=1000000
        )


if __name__ == "__main__":
    main()
