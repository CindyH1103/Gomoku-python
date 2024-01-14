# basic setting
board_size = 9
n_in_row = 5
init_lr = 2e-3
temp = 1.0
n_playout = 600
c_puct = 5
buffer_size = 5000  # 5000 for prioritized replay buffer, 10000 for uniform replay buffer
batch_size = 512  # 512 for pure/res, 256 for gomokuNet
play_batch_size = 1
epochs = 3
kl_targ = 0.02
check_freq = 30  # 50 for most of the training
game_batch_num = 1500
pure_mcts_playout_num = 1000
prob_compete_with_best = 0.05
seed = 1

# model related setting
exp_name = "finetune_PC"  # should be train/finetune_PC/finetune_PC_FC/finetune_FCs
model_type = "res"  # pure/res/gomokuNet
feature_channel = 256  # 128 for gomokuNet, 256 for pure/res
num_res = 5
init_model = 'best_policy_finetune_PC_res.model'

# finetune related
beta = 2.0
lamb = 1.0

# replay buffer setting
priority_replay = True
replay_alpha = 0.8
replay_e = 0.001
replay_beta = 0.6

# gpu setting
gpu_id = "5"
use_gpu = True