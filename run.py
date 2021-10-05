import os
import json
import torch
import pprint
import numpy as np
from tensorboard_logger import Logger as TbLogger
import warnings
from options import get_options

from problems.problem_tsp import TSP
from problems.problem_vrp import CVRP
from agent.ppo import PPO

def load_agent(name):
    agent = {
        'ppo': PPO,
    }.get(name, None)
    assert agent is not None, "Currently unsupported agent: {}!".format(name)
    return agent

def load_problem(name):
    problem = {
        'tsp': TSP,
        'vrp': CVRP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def run(opts):

    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tb and not opts.distributed:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    # Save arguments so exact configuration can always be found
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    # Figure out what's the problem
    problem = load_problem(opts.problem)(
                            p_size = opts.graph_size,
                            step_method = opts.step_method,
                            init_val_met = opts.init_val_met,
                            with_assert = opts.use_assert,
                            P = opts.P,
                            DUMMY_RATE = opts.dummy_rate)
    
    # Figure out the RL algorithm
    agent = load_agent(opts.RL_agent)(problem.NAME, problem.size,  opts)

    # Load data from load_path
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        agent.load(load_path)
    
    
    # Do validation only
    if opts.eval_only:
        # Load the validation datasets
        agent.start_inference(problem, opts.val_dataset, tb_logger)
        
    else:
        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            agent.opts.epoch_start = epoch_resume + 1
    
        # Start the actual training loop
        agent.start_training(problem, opts.val_dataset, tb_logger)            


if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run(get_options())
