### STAT 946 Supplementary Material for: 
## Student Teacher DQN for Reinforcement Learning

## Code Breakdown
The DQN and A3C models that were used (vanilla) were implemented as-in [Tensorpack]((https://github.com/ppwwyyxx/tensorpack/tree/master/)) for the Atari games.

The model outlined in ``student_dqn.py`` is the from scratch development of the student ST-DQN which was derived in the paper. The model outlined in ``student_teach_dqn.py`` is the attempted memory efficient model derived by editing the tensorpack implementation. This implementation is potentially incomplete.

``train-imitation.py`` includes the files which correspond to the imitation model that was trained. 

## Gameplay Videos
Results for the baseline model and the imitation model are included. The ST-DQN is not due to the inability to reach desirable performance, as discussed in the paper, on the atari games. Results for the cartpole task could be included, but the clips are rather un-interesting. 

### Baseline Teaching Model (Tensorpack Implementation)
![Imitation Model](Teacher%20Baseline.gif)

### Best Results from Imitation Model
![Imitation Model](Imitation%20Best.gif)
