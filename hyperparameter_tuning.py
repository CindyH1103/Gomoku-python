import ray
from ray import tune, train
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune import CLIReporter
import torch
from train import TrainPipeline, set_seed
from config import *
import os
import time


def train_tune(config):
    pv_ratio = config['pv_ratio']
    training_pipeline = TrainPipeline(init_lr, '/home/huangyixin/AI/best_policy_train_pure.model', pv_ratio=pv_ratio)
    for i in range(50):
        training_pipeline.collect_selfplay_data(play_batch_size)
        if len(training_pipeline.data_buffer) > batch_size:
            training_pipeline.policy_update()

    win_ratio = training_pipeline.policy_evaluate_for_tune()
    # ray.air.session.report(win_ratio)
    tune.report({"win_ratio": win_ratio})
    # return win_ratio


ray.init(num_gpus=2)
# - 定义搜索空间
search_space = {"pv_ratio": tune.uniform(0.5, 2)}
set_seed()
os.environ['CUDA_VISIBLE_DEVICE'] = "1, 3"
# reporter = CLIReporter(
#     metric_columns=["win_ratio"],
# )

# - 配置Ray Tune
tuner = tune.Tuner(
    tune.with_resources(train_tune, {"gpu": 2}),
    tune_config=tune.TuneConfig(
        metric="_metric/win_ratio",
        mode="max",
        scheduler=AsyncHyperBandScheduler(),
        num_samples=10,
    ),
    param_space=search_space,
)

# - 运行Ray Tune
results = tuner.fit()
print("Best hyperparameters found were: ", results.get_best_result().config)

print(results)
best_trial = results.get_best_result(metric="_metric/win_ratio", mode="max")
print(best_trial.config["pv_ratio"])
