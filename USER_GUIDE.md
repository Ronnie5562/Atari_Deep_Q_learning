# Complete User Guide - DQN Breakout Project

This guide will walk you through everything you need to know about training and playing a DQN agent with this project.

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or run the setup script
python setup.py

# Accept Atari ROM license
AutoROM --accept-license
```

### 2. Train the Agent

```bash
python train.py
```

### 3. Play with the Agent

```bash
python play.py
```

## Project Structure

```
dqn-breakout/
├── train.py              # Main training script
├── play.py               # Playing and evaluation script
├── config.py             # Configuration file (MODIFY THIS)
├── run_experiment.py     # Easy experiment runner
├── setup.py              # Installation script
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
├── USER_GUIDE.md         # This guide
├── dqn_model.zip         # Trained model (generated)
├── logs/                 # Training logs
└── experiments/          # Experiment results
```

## How to Change Hyperparameters

### Method 1: Modify config.py (RECOMMENDED)

Open `config.py` and modify the `CUSTOM_HYPERPARAMS` section:

```python
CUSTOM_HYPERPARAMS = {
    'name': 'My Custom Set',
    'learning_rate': 2e-4,     # Change this
    'gamma': 0.98,             # Change this
    'batch_size': 64,          # Change this
    'epsilon_start': 1.0,      # Change this
    'epsilon_end': 0.02,       # Change this
    'epsilon_decay': 0.15,     # Change this
    'description': 'My custom configuration'
}
```

Then run:

```bash
python run_experiment.py
# Choose option 1 for custom experiment
```

### Method 2: Add to Predefined Sets

In `config.py`, add your configuration to `HYPERPARAMETER_SETS`:

```python
HYPERPARAMETER_SETS = [
    # ... existing sets ...
    {
        'name': 'My New Set',
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.1,
        'description': 'My experimental configuration'
    }
]
```

### Method 3: Direct Modification in train.py

Find the hyperparameter dictionaries in `train.py` and modify them directly.

## 🔧 Hyperparameter Explanation

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|---------------|---------|
| `learning_rate` | How fast the agent learns | 1e-5 to 1e-3 | Higher = faster learning, more unstable |
| `gamma` | Discount factor for future rewards | 0.9 to 0.999 | Higher = more long-term focus |
| `batch_size` | Number of experiences per update | 16 to 128 | Higher = more stable updates |
| `epsilon_start` | Initial exploration rate | 0.8 to 1.0 | Higher = more initial exploration |
| `epsilon_end` | Final exploration rate | 0.01 to 0.1 | Lower = more exploitation |
| `epsilon_decay` | Fraction of training for exploration | 0.05 to 0.3 | Higher = longer exploration |

## Getting the ZIP Policy

The **dqn_model.zip** file is automatically created when you train the agent:

### Where it's created

1. **Main training**: `dqn_model.zip` in the project root
2. **Experiments**: `experiments/[experiment_name]/model.zip`

### What it contains

- Trained neural network weights
- Model architecture
- Training configuration
- Preprocessing parameters

### How to use it

```python
from stable_baselines3 import DQN

# Load the model
model = DQN.load("dqn_model.zip")

# Use for prediction
action, _ = model.predict(observation)
```

## Running Experiments

### Option 1: Individual Experiment

```bash
python run_experiment.py
# Choose option 1 for custom experiment
```

### Option 2: All Experiments

```bash
python run_experiment.py
# Choose option 2 to run all predefined experiments
```

### Option 3: Specific Experiment

```bash
python run_experiment.py
# Choose option 3 and select an experiment by index
```

### Option 4: Quick Test

```bash
python run_experiment.py
# Choose option 4 for quick testing (reduced training time)
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
```

Open <http://localhost:6006> to view:

- Loss curves
- Reward trends
- Episode lengths
- Learning progress

### Console Output

The training script shows:

- Current episode
- Mean reward
- Episode length
- Exploration rate

## Playing with the Agent

### Visual Playback

```bash
python play.py
# Choose option 1 for visual playback
```

### Performance Evaluation

```bash
python play.py
# Choose option 2 for performance evaluation
```

### Detailed Statistics

```bash
python play.py
# Choose option 3 for detailed statistics
```

## 📋 Hyperparameter Tuning Table

For your assignment, use this table format:

| Hyperparameter Set | Learning Rate | Gamma | Batch Size | Epsilon Start | Epsilon End | Epsilon Decay | Observed Behavior |
|-------------------|---------------|--------|------------|---------------|-------------|---------------|-------------------|
| lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | [Your observations] |
| lr=5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 5e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | [Your observations] |

## 🔍 Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Solution: Train the model first with `python train.py`

2. **GPU memory issues**
   - Solution: Reduce batch size or use CPU

3. **Slow training**
   - Solution: Use GPU, reduce total timesteps, or use quick mode

4. **Poor performance**
   - Solution: Train longer, adjust hyperparameters, or use CNN policy

5. **Installation issues**
   - Solution: Run `python setup.py` or install dependencies manually

### Performance Tips

1. **Use CNN Policy** for Atari games (not MLP)
2. **Train for at least 1M steps** for good performance
3. **Use frame stacking** (4 frames) for temporal information
4. **Monitor TensorBoard** for training progress
5. **Evaluate periodically** to track improvement

## Assignment Checklist

- [ ] Install all dependencies
- [ ] Train DQN agent with CNN policy
- [ ] Compare CNN vs MLP policy (note: MLP will fail)
- [ ] Test at least 5 different hyperparameter sets
- [ ] Record results in the provided table format
- [ ] Save trained model (dqn_model.zip)
- [ ] Create play.py script that loads and runs the agent
- [ ] Record video of agent playing
- [ ] Document hyperparameter tuning results
- [ ] Create GitHub repository with all files
- [ ] Write README with results and observations

## Creating Video Documentation

1. **Record gameplay**:

   ```bash
   python play.py
   # Choose visual playback and record screen
   ```

2. **Show training progress**:
   - Screenshot of TensorBoard
   - Console output during training
   - Final model performance

3. **Demonstrate hyperparameter effects**:
   - Show different agents trained with different parameters
   - Compare their performance visually

## Additional Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)
- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [TensorBoard Guide](https://tensorboard.dev/)
