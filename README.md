# DQN Agent for Breakout - Deep Q-Learning Assignment

This project implements a Deep Q-Network (DQN) agent to play the Atari game Breakout using Stable Baselines3 and Gymnasium.

## Environment
- **Game**: ALE/Breakout-v5
- **Framework**: Stable Baselines3 with Gymnasium
- **Policy**: CNN Policy (Convolutional Neural Network)

## Files Structure
```
Atari_Deep_Q_learning/
    ├── train.py              # Training script
    ├── play.py               # Playing/evaluation script
    ├── requirements.txt      # Dependencies
    ├── README.md            # This file
    ├── dqn_model.zip        # Trained model (generated after training)
    └── logs/                # Training logs and tensorboard data
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Ronnie5562/Atari_Deep_Q_learning.git
cd Atari_Deep_Q_learning
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Accept ROM license** (required for Atari games)
```bash
AutoROM --accept-license
```

## Usage

### Training the Agent

Run the training script:
```bash
python train.py
```

The script will prompt you to choose:
1. **Policy comparison**: Compare CNN vs MLP policies
2. **Hyperparameter tuning**: Test different hyperparameter configurations
3. **Single training run**: Train with default parameters

### Playing with the Trained Agent

After training, run the playing script:
```bash
python play.py
```

The script offers three modes:
1. **Visual playback**: Watch the agent play with GUI
2. **Performance evaluation**: Evaluate without visualization
3. **Detailed statistics**: Play with comprehensive statistics

## Hyperparameter Tuning Results

The following hyperparameter configurations were tested:

| Hyperparameter Set | Learning Rate | Gamma | Batch Size | Epsilon Start | Epsilon End | Epsilon Decay | Observed Behavior |
|-------------------|---------------|--------|------------|---------------|-------------|---------------|-------------------|
| Set 1 - Default | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | Baseline performance with stable learning |
| Set 2 - Higher LR | 5e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | Faster initial learning but more unstable |
| Set 3 - Lower Gamma | 1e-4 | 0.95 | 32 | 1.0 | 0.01 | 0.1 | Less focus on long-term rewards |
| Set 4 - Larger Batch | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.1 | More stable updates, slower convergence |
| Set 5 - Slower Decay | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.2 | Extended exploration phase |

### Key Findings:
- **CNN Policy** significantly outperforms MLP Policy for Atari environments
- **Default parameters** (Set 1) provide good baseline performance
- **Higher learning rate** (Set 2) can speed up training but may cause instability
- **Lower gamma** (Set 3) reduces long-term planning ability
- **Larger batch size** (Set 4) provides more stable gradients
- **Slower epsilon decay** (Set 5) maintains exploration longer

## Model Architecture

The DQN agent uses:
- **Policy**: CNN Policy (Convolutional Neural Network)
- **Input**: 84x84x4 stacked grayscale frames
- **Output**: Q-values for 4 possible actions
- **Replay Buffer**: 100,000 experiences
- **Target Network**: Updated every 10,000 steps

## Training Configuration

**Default Hyperparameters:**
- Learning Rate: 1e-4
- Gamma (discount factor): 0.99
- Batch Size: 32
- Epsilon Start: 1.0
- Epsilon End: 0.01
- Epsilon Decay: 0.1 (fraction of training)
- Training Steps: 1,000,000
- Buffer Size: 100,000

## Getting the ZIP Policy

The **dqn_model.zip** file is automatically generated when you run the training script. This file contains:
- The trained neural network weights
- Model architecture information
- Training parameters

### Where to find it:
1. After running `python train.py`, the model is saved as `dqn_model.zip` in the project directory
2. The file is automatically loaded by `play.py` to run the trained agent

## Video Demonstration

[Watch Here](https://github.com/user-attachments/assets/6b8ba19f-fbd9-4789-bde0-1777221410dd)

## Performance Metrics

The agent's performance can be evaluated using:
- **Average reward per episode**
- **Episode length**
- **Success rate** (completing levels)
- **Learning curve** (reward over training time)

Check the `logs/` directory for detailed training metrics and TensorBoard logs.

## Troubleshooting

### Common Issues:

1. **ImportError**: Install all dependencies with `pip install -r requirements.txt`
2. **ROM not found**: Run `AutoROM --accept-license`
3. **CUDA errors**: Set `device='cpu'` in the DQN model if GPU issues occur
4. **Display issues**: Ensure you have a display available for rendering

### System Requirements:
- Python 3.8+
- 4GB+ RAM
- GPU recommended for faster training
- Display for visualization

## Advanced Usage

### Custom Hyperparameters

To test custom hyperparameters, modify the `hyperparameter_sets` list in `train.py`:

```python
hyperparameter_sets = [
    {
        'name': 'Custom Set',
        'learning_rate': 2e-4,  # Your custom learning rate
        'gamma': 0.98,          # Your custom gamma
        'batch_size': 64,       # Your custom batch size
        'epsilon_start': 1.0,   # Your custom epsilon start
        'epsilon_end': 0.02,    # Your custom epsilon end
        'epsilon_decay': 0.15   # Your custom epsilon decay
    }
]
```

### Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## License

This project is for educational purposes.
