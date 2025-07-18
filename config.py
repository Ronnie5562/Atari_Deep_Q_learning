"""
Configuration file for DQN Breakout training
Modify the hyperparameters here to test different configurations
"""

# Environment settings
ENV_NAME = "BreakoutNoFrameskip-v4"
SEED = 42

# Training settings
TOTAL_TIMESTEPS = 1000000
EVAL_FREQ = 10000
LEARNING_STARTS = 50000
BUFFER_SIZE = 100000
TARGET_UPDATE_INTERVAL = 10000
TRAIN_FREQ = 4

# Model save settings
MODEL_SAVE_PATH = "dqn_model.zip"
LOG_DIR = "./logs"

# Default hyperparameters - MODIFY THESE TO TEST DIFFERENT CONFIGURATIONS
DEFAULT_HYPERPARAMS = {
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'batch_size': 32,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.1
}

# Hyperparameter sets for experimentation
# Add your own configurations here
HYPERPARAMETER_SETS = [
    {
        'name': 'Set 1 - Default',
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.1,
        'description': 'Standard DQN parameters for Atari'
    },
    {
        'name': 'Set 2 - Higher Learning Rate',
        'learning_rate': 5e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.1,
        'description': 'Faster learning with higher LR'
    },
    {
        'name': 'Set 3 - Lower Gamma',
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.1,
        'description': 'Less focus on long-term rewards'
    },
    {
        'name': 'Set 4 - Larger Batch Size',
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'batch_size': 64,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.1,
        'description': 'More stable gradient updates'
    },
    {
        'name': 'Set 5 - Slower Epsilon Decay',
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.2,
        'description': 'Extended exploration phase'
    },
    {
        'name': 'Set 6 - Aggressive Learning',
        'learning_rate': 2e-4,
        'gamma': 0.99,
        'batch_size': 64,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.05,
        'description': 'Fast learning with quick exploration decay'
    },
    {
        'name': 'Set 7 - Conservative Learning',
        'learning_rate': 5e-5,
        'gamma': 0.995,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.02,
        'epsilon_decay': 0.15,
        'description': 'Slow, stable learning with extended exploration'
    }
]

# Policy types to compare
POLICY_TYPES = ["CnnPolicy", "MlpPolicy"]

# Evaluation settings
EVAL_EPISODES = 10
PLAY_EPISODES = 3

# Rendering settings
RENDER_MODE = "human"
RENDER_DELAY = 0.01  # seconds between frames during play

# TensorBoard logging
USE_TENSORBOARD = True
TENSORBOARD_LOG = "./logs/tensorboard"

# Performance tracking
TRACK_REWARDS = True
TRACK_EPISODE_LENGTH = True
TRACK_ACTIONS = True

# Quick training mode (for testing)
QUICK_MODE = False
QUICK_TIMESTEPS = 100000

def get_hyperparams_by_name(name):
    """Get hyperparameters by set name"""
    for params in HYPERPARAMETER_SETS:
        if params['name'] == name:
            return {k: v for k, v in params.items() if k not in ['name', 'description']}
    return DEFAULT_HYPERPARAMS

def get_hyperparams_by_index(index):
    """Get hyperparameters by index"""
    if 0 <= index < len(HYPERPARAMETER_SETS):
        params = HYPERPARAMETER_SETS[index]
        return {k: v for k, v in params.items() if k not in ['name', 'description']}
    return DEFAULT_HYPERPARAMS

def list_hyperparameter_sets():
    """List all available hyperparameter sets"""
    print("Available hyperparameter sets:")
    for i, params in enumerate(HYPERPARAMETER_SETS):
        print(f"{i}: {params['name']} - {params['description']}")

def validate_hyperparams(hyperparams):
    """Validate hyperparameter values"""
    errors = []

    if not (1e-6 <= hyperparams['learning_rate'] <= 1e-1):
        errors.append("Learning rate should be between 1e-6 and 1e-1")

    if not (0.8 <= hyperparams['gamma'] <= 0.999):
        errors.append("Gamma should be between 0.8 and 0.999")

    if not (8 <= hyperparams['batch_size'] <= 256):
        errors.append("Batch size should be between 8 and 256")

    if not (0.5 <= hyperparams['epsilon_start'] <= 1.0):
        errors.append("Epsilon start should be between 0.5 and 1.0")

    if not (0.01 <= hyperparams['epsilon_end'] <= 0.2):
        errors.append("Epsilon end should be between 0.01 and 0.2")

    if not (0.05 <= hyperparams['epsilon_decay'] <= 0.5):
        errors.append("Epsilon decay should be between 0.05 and 0.5")

    if errors:
        print("Hyperparameter validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True

# Custom hyperparameter set - MODIFY THIS TO TEST YOUR OWN CONFIGURATIONS
CUSTOM_HYPERPARAMS = {
    'name': 'Custom Set',
    'learning_rate': 1e-4,     # Try: 5e-5, 1e-4, 2e-4, 5e-4
    'gamma': 0.99,             # Try: 0.95, 0.98, 0.99, 0.995
    'batch_size': 32,          # Try: 16, 32, 64, 128
    'epsilon_start': 1.0,      # Try: 0.8, 1.0
    'epsilon_end': 0.01,       # Try: 0.01, 0.02, 0.05
    'epsilon_decay': 0.1,      # Try: 0.05, 0.1, 0.15, 0.2
    'description': 'Your custom configuration'
}