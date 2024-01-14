## Folder Structure
```
├── best_policy_finetune_PC_res.model		# the residual block based network checkpoint 
├── best_policy_finetune_PC_pure.model		# the convolution layer based network checkpoint
├── best_policy_finetune_gomokuNet.model	# the GomokuNet based network checkpoint
├── config.py                       		# the file that contains all hyperparameters needed for training
├── game.py 								# the file used for simulating Gomoku
├── hyperparameter_tuning.py           		# the file that implements hyperparameter tuning by ray.tune
├── mcts_alphaZero.py     					# the mcts AI agent tailored for AlphaZero
├── mcts_pure.py                    		# the mcts agent for pure MCTS player
├── model.py              				 	# the file that contains the implementation for all 3 policy value networks
├── mydraw.py         						# the file used for game visualization
├── pbt.py    								# the file that implements a simple population based training using torch.multiprocessing
├── sumTree.py                    			# the data structure used in priority replay buffer
└── train.py								# the scripts for training
```

### Environment Setting

Since the CUDA on the server is 11.0, I choose `torch==1.12.1` to run the experiment. Theoretically a higher version is compatible.

    conda create -n gomoku python=3.9
    source activate gomoku
    pip install torch==1.12.1
    pip install einops==0.7.0
    pip install numpy==1.24.3
    pip install ray==2.6.0

### Training

run in the root directory. The model file will be saved under the root directory and the experiment will be saved in `./log` automatically

    python train.py

### Training Hyperparameter Setting

All changeable hyperparameters are included in `config.py`. Some key hyperparameters will be explained as follows:

* `c_puct` : default to be 5, important hyperparameter MCTS search tree
* `buffer_size`: default to be 10000, the buffer size of the data buffer
* `batch_size`: default to be 512
* `check_freq`: default to be 20, the frequency to check and possibly update the saved model file 
* `pure_mcts_playout_num`: default to be 1000, the initial setting of the pure MCTS player 
* `exp_name`: default to be `'finetune_PC'`, could be chosen from `['train', 'finetune_PC', 'finetune_PC_FC', 'finetune_FC']`. The loss setting. If `'train'` is selected, only policy and value loss will be used. If `'finetune_PC'` is chosen, path consistency loss will be added. If `'finetune_FC'` is picked, feature consistency loss will be included. And `'finetune_PC_FC'` will bring a combination of both.
* `model_type`: default to be `'res'`, could be chosen from `['pure', 'res', 'gomokuNet']`. The policy value network architecture
* `feature_channel`: default to be 256. If the trained pure/res network is used, 256 should be set. If the trained gomokuNet is utilized, 128 should be set. The feature channel of the policy value network.
* `num_res`: default to be 5, the number of residual blocks stacked in the residual block based policy value network
* `init_model`: default to be None. The checkpoint to be reloaded for training/finetuning. The provided model should be compatible with the above model setting
* `beta`: default to be 2, the coefficient of path consistency loss in the loss function
* `lamba`: default to be 1, the coefficient of feature consistency loss in the loss function
* `priority _replay`: whether to use the priority replay buffer. If False is set, a uniform replay buffer will be selected instead.
* `gpu_id`: the GPU to be used in the experiment
* `use_gpu`: whether to use GPU in the training/finetuning

### Start a Game!

To start a game whether between 2 AI agents or between AI agent and human player, we could just add some simple modifications to `game.py`. The following part should be kept unchanged to correctly set up the environment and start a game.

```python
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    board_width = 9
    board_height = 9
    n_in_row = 5
    board = Board(width=board_width,
                  height=board_height,
                  n_in_row=n_in_row,
                  device=torch.device('cuda'))
    task = Game(board, device=torch.device('cuda'))
```

#### Between 2 AlphaZero-based AI agents

Matches between 2 AI agents could be achieved by the `policy_compete1` function in the class Game. If GomokuNet is used as one of the competitor, the corresponding feature channel should be set up correctly to be compatible with the provided model file. The complete code for this function could be easily found in `game.py`.

```python
win_ratio = task.policy_compete1({model_file1}, {model_file2}, {model_type1}, {model_type2})
```

#### Between AlphaZero-based AI agent and human player

It is worth noting that pygame could not be displayed on the server connected through a VSCode/pycharm terminal and we've only tested the visualization module on our PC.

```python
task.human_play({model_file}, {model_type})
```

#### Between Pure MCTS player and AlphaZero-based AI agents

The match between baseline player could be realized through the `policy_evaluate` function. Also, all model hyperparameters need to be set to be compatible with the model file selected.

```python
_, poses, winners = task.policy_evaluate(checkpoint={model_file}, model_type={model_type}, pure_mcts_playout_num=200, n_games=10)
```

To further visualize the game, we could simply add the line below:

```python
mydraw._draw(poses, winners, board_width)
```

We uploaded all trained models to show our progress. The best model is `best_policy_finetune_PC_res.model` and should be used for testing if only one model is to be selected.  The model with the fastest speed is `best_policy_finetune_PC_pure.model`

