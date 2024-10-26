# RL Assignment

Contributors: Willem Van Der Merwe

### Project Overview
This project explores reinforcement learning (RL) to tackle [briefly describe the problem, e.g., "grid optimization," "game environment," "resource allocation," etc.]. The project applies RL techniques to enable an agent to [describe goal, e.g., "maximize rewards," "minimize penalties," etc.] in a dynamic environment.

### Key Components
Environment: [e.g., Custom Gym environment, Grid2Op, etc.]
RL Algorithms: [List algorithms used, e.g., Q-Learning, PPO]
Performance Metrics: [e.g., Cumulative reward, episode length, convergence speed]
### Installation
#### Clone this repository:
`git clone https://github.com/username/repository.git`
#### Install the dependencies:
Conda environment file included

`conda env create`
### To train the agent:
`python train_agent.py --algorithm PPO`
### To evaluate performance:
`python evaluate_agent.py --episodes 100`
### Folder Structure
run.py: Script for training and inferencing the agent. 
/configs: Configuration files for different environments and algorithms.
/results: Stores logs, model checkpoints, and performance graphs.
### Results
To view results please run `tensorboard --logdir ./tensorb_run2` which is the last run I included ./tensorb_run1 
and ./tensor_board directories which include convergence issues due to un optimized implementations, which shows
how things improved along the way

