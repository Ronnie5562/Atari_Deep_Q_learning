#!/usr/bin/env python3
"""
Easy experiment runner for DQN Breakout
This script makes it easy to run different hyperparameter configurations
"""

import os
import sys
import time
import json
from datetime import datetime
from config import *
from train import train_dqn_agent, create_environment
from play import evaluate_agent_performance

def run_single_experiment(hyperparams, experiment_name, timesteps=None):
    """Run a single experiment with given hyperparameters"""

    print(f"\n{'='*50}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*50}")

    # Display hyperparameters
    print("Hyperparameters:")
    for key, value in hyperparams.items():
        if key not in ['name', 'description']:
            print(f"  {key}: {value}")

    # Validate hyperparameters
    if not validate_hyperparams(hyperparams):
        print("Invalid hyperparameters. Skipping experiment.")
        return None

    # Set timesteps
    if timesteps is None:
        timesteps = QUICK_TIMESTEPS if QUICK_MODE else TOTAL_TIMESTEPS

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{experiment_name}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Save hyperparameters
    with open(f"{experiment_dir}/hyperparams.json", 'w') as f:
        json.dump(hyperparams, f, indent=2)

    # Train the model
    start_time = time.time()

    try:
        model, log_dir = train_dqn_agent(
            hyperparams,
            policy_type="CnnPolicy",
            total_timesteps=timesteps
        )

        training_time = time.time() - start_time

        # Save model with experiment name
        model_path = f"{experiment_dir}/model.zip"
        model.save(model_path)

        # Evaluate the model
        print(f"\nEvaluating model...")
        env = create_environment()
        rewards, episode_lengths = [], []

        for episode in range(EVAL_EPISODES):
            obs = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                steps += 1

                if done[0]:
                    break

            rewards.append(total_reward)
            episode_lengths.append(steps)

        env.close()

        # Calculate statistics
        results = {
            'experiment_name': experiment_name,
            'hyperparams': hyperparams,
            'training_time': training_time,
            'timesteps': timesteps,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'best_reward': float(max(rewards)),
            'worst_reward': float(min(rewards)),
            'eval_episodes': EVAL_EPISODES,
            'log_dir': log_dir,
            'model_path': model_path
        }

        # Save results
        with open(f"{experiment_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n Experiment completed!")
        print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Results saved to: {experiment_dir}")

        return results

    except Exception as e:
        print(f"Experiment failed: {e}")
        return None

def run_all_experiments():
    """Run all predefined hyperparameter experiments"""

    print("🚀 Running all hyperparameter experiments...")

    results = []

    for i, params in enumerate(HYPERPARAMETER_SETS):
        experiment_name = f"exp_{i+1}_{params['name'].replace(' ', '_').replace('-', '_')}"

        # Extract hyperparameters (exclude name and description)
        hyperparams = {k: v for k, v in params.items() if k not in ['name', 'description']}

        result = run_single_experiment(hyperparams, experiment_name)

        if result:
            results.append(result)

        # Small break between experiments
        time.sleep(5)

    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"experiments/comparison_{timestamp}.json"

    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print_comparison_table(results)

    return results

def print_comparison_table(results):
    """Print a comparison table of all results"""

    if not results:
        print("No results to compare.")
        return

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON TABLE")
    print(f"{'='*80}")

    # Header
    print(f"{'Experiment':<20} {'Mean Reward':<12} {'Std Reward':<12} {'Best Reward':<12} {'Training Time':<15}")
    print("-" * 80)

    # Sort by mean reward
    sorted_results = sorted(results, key=lambda x: x['mean_reward'], reverse=True)

    for result in sorted_results:
        print(f"{result['experiment_name']:<20} "
              f"{result['mean_reward']:<12.2f} "
              f"{result['std_reward']:<12.2f} "
              f"{result['best_reward']:<12.2f} "
              f"{result['training_time']:<15.2f}")

    print("-" * 80)
    print(f"Best performing experiment: {sorted_results[0]['experiment_name']}")
    print(f"Mean reward: {sorted_results[0]['mean_reward']:.2f}")

def run_custom_experiment():
    """Run experiment with custom hyperparameters"""

    print("🔧 Running custom experiment...")

    # Use custom hyperparameters from config
    hyperparams = {k: v for k, v in CUSTOM_HYPERPARAMS.items() if k not in ['name', 'description']}

    result = run_single_experiment(hyperparams, "custom_experiment")

    return result

def main():
    """Main function"""

    print("DQN Breakout Experiment Runner")
    print("=" * 50)

    # Create experiments directory
    os.makedirs("experiments", exist_ok=True)

    # Import numpy for calculations
    global np
    import numpy as np

    print("\nChoose experiment type:")
    print("1. Run single experiment (custom hyperparameters)")
    print("2. Run all predefined experiments")
    print("3. Run specific experiment by index")
    print("4. Quick mode (reduced training time)")

    choice = input("\nEnter your choice (1-4): ")

    if choice == "1":
        result = run_custom_experiment()

    elif choice == "2":
        results = run_all_experiments()

    elif choice == "3":
        list_hyperparameter_sets()
        index = int(input("\nEnter experiment index: "))

        if 0 <= index < len(HYPERPARAMETER_SETS):
            params = HYPERPARAMETER_SETS[index]
            hyperparams = {k: v for k, v in params.items() if k not in ['name', 'description']}
            experiment_name = f"exp_{params['name'].replace(' ', '_').replace('-', '_')}"

            result = run_single_experiment(hyperparams, experiment_name)
        else:
            print("Invalid index!")

    elif choice == "4":
        print("🚀 Enabling quick mode...")
        global QUICK_MODE
        QUICK_MODE = True

        # Run a quick experiment
        hyperparams = get_hyperparams_by_index(0)  # Use first set
        result = run_single_experiment(hyperparams, "quick_test", QUICK_TIMESTEPS)

    else:
        print("Invalid choice. Running custom experiment...")
        result = run_custom_experiment()

    print("\n Experiment runner completed!")
    print("Check the 'experiments' directory for detailed results.")

if __name__ == "__main__":
    main()
