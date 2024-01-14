import argparse
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as _mp
from train import TrainPipeline
from config import *

mp = _mp.get_context('spawn')


def exploit_and_explore(top_checkpoint_path, bot_checkpoint_path, hyper_params,
                        perturb_factors=(1.2, 0.8)):
    """Copy parameters from the better model and the hyperparameters
       and running averages from the corresponding optimizer."""
    # Copy model parameters
    checkpoint = torch.load(top_checkpoint_path)
    state_dict = checkpoint['net']
    optimizer_state_dict = checkpoint['optim']
    for hyperparam_name in hyper_params['optimizer']:
        perturb = np.random.choice(perturb_factors)
        for param_group in optimizer_state_dict['param_groups']:
            param_group[hyperparam_name] *= perturb

    checkpoint = dict(model_state_dict=state_dict,
                      optim_state_dict=optimizer_state_dict,
                      batch_size=batch_size)
    torch.save(checkpoint, bot_checkpoint_path)


class Worker(mp.Process):
    def __init__(self, epoch, max_epoch, population, finish_tasks):
        super().__init__()
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.population = population
        self.lr = np.random.choice(np.logspace(-5, 0, base=10))
        self.finish_tasks = finish_tasks

    def run(self):
        while True:
            if self.epoch.value > self.max_epoch:
                break
            # Train
            task = self.population.get()
            # self.trainer.set_id(task['id'])
            checkpoint_path = "current_policy" + str(task['id']) + "_" + exp_name + "_" + model_type + ".model"
            if os.path.isfile(checkpoint_path):
                self.trainer = TrainPipeline(self.lr, checkpoint_path)
            else:
                self.trainer = TrainPipeline(self.lr, None)
            try:
                score = self.trainer.train_tune(task['id'])
                self.finish_tasks.put(dict(id=task['id'], score=score))
            except KeyboardInterrupt:
                break


class Explorer(mp.Process):
    def __init__(self, epoch, max_epoch, population, finish_tasks, hyper_params):
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.hyper_params = hyper_params

    def run(self):
        while True:
            if self.epoch.value > self.max_epoch:
                break
            if self.population.empty() and self.finish_tasks.full():
                print("Exploit and explore")
                tasks = []
                while not self.finish_tasks.empty():
                    tasks.append(self.finish_tasks.get())
                tasks = sorted(tasks, key=lambda x: x['score'], reverse=True)
                print('Best score on', tasks[0]['id'], 'is', tasks[0]['score'])
                print('Worst score on', tasks[-1]['id'], 'is', tasks[-1]['score'])
                fraction = 0.2
                cutoff = int(np.ceil(fraction * len(tasks)))
                tops = tasks[:cutoff]
                bottoms = tasks[len(tasks) - cutoff:]
                for bottom in bottoms:
                    top = np.random.choice(tops)
                    top_checkpoint_path = "current_policy" + str(top['id']) + "_" + exp_name + "_" + model_type + ".model"
                    bot_checkpoint_path = "current_policy" + str(bottom['id']) + "_" + exp_name + "_" + model_type + ".model"
                    exploit_and_explore(top_checkpoint_path, bot_checkpoint_path, self.hyper_params)
                    with self.epoch.get_lock():
                        self.epoch.value += 1
                for task in tasks:
                    self.population.put(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--gpu_id", type=str, default='6', help="")
    parser.add_argument("--population_size", type=int, default=10, help="")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu_id
    # mp.set_start_method("spawn")
    mp = mp.get_context('forkserver')

    population_size = args.population_size
    max_epoch = 1000
    pathlib.Path('checkpoints').mkdir(exist_ok=True)
    population = mp.Queue(maxsize=population_size)
    finish_tasks = mp.Queue(maxsize=population_size)
    epoch = mp.Value('i', 0)
    for i in range(population_size):
        population.put(dict(id=i, score=0))
    hyper_params = {'optimizer': ["lr"]}

    workers = [Worker(epoch, max_epoch, population, finish_tasks) for _ in range(3)]
    workers.append(Explorer(epoch, max_epoch, population, finish_tasks, hyper_params))
    print("begin search!!!")
    [w.start() for w in workers]
    [w.join() for w in workers]
    task = []
    while not finish_tasks.empty():
        task.append(finish_tasks.get())
    while not population.empty():
        task.append(population.get())
    task = sorted(task, key=lambda x: x['score'], reverse=True)
    print('best score on', task[0]['id'], 'is', task[0]['score'])
