import os
import time
import argparse
import torch


def get_options(args=None):
    
    parser = argparse.ArgumentParser(description="Dual-Aspect Collaborative Transformer")

    # Overall settings
    parser.add_argument('--problem', default='tsp', choices = ['vrp', 'tsp'], help="the targeted problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="the number of customers in the targeted problem (graph size)")
    parser.add_argument('--dummy_rate', type=float, default=0.5, help="add DUMMY_RATE * graph_size nodes as dummy depots (for CVRP only)")
    parser.add_argument('--step_method', default='2_opt', choices = ['2_opt','swap','insert'])
    parser.add_argument('--init_val_met', choices = ['random','greedy','seq'], default = 'greedy', help='method to generate initial solutions for inference')
    parser.add_argument('--no_cuda', action='store_true', help='disable GPUs')
    parser.add_argument('--no_tb', action='store_true', help='disable Tensorboard logging')
    parser.add_argument('--show_figs', action='store_true', help='enable figure logging')
    parser.add_argument('--no_saving', action='store_true', help='disable saving checkpoints')
    parser.add_argument('--use_assert', action='store_true', help='enable assertion')
    parser.add_argument('--no_DDP', action='store_true', help='disable distributed parallel')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use')
    
    # DACT parameters
    parser.add_argument('--v_range', type=float, default=6., help='to control the entropy')
    parser.add_argument('--DACTencoder_head_num', type=int, default=4, help='head number of DACT encoder')
    parser.add_argument('--DACTdecoder_head_num', type=int, default=4, help='head number of DACT decoder')
    parser.add_argument('--critic_head_num', type=int, default=6, help='head number of critic encoder')
    parser.add_argument('--embedding_dim', type=int, default=64, help='dimension of input embeddings (NEF & PFE)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='number of stacked layers in the encoder')
    parser.add_argument('--normalization', default='layer', help="normalization type, 'layer' (default) or 'batch'")

    # Training parameters
    parser.add_argument('--RL_agent', default='ppo', choices = ['ppo'], help='RL Training algorithm')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor for future rewards')
    parser.add_argument('--K_epochs', type=int, default=3, help='mini PPO epoch')
    parser.add_argument('--eps_clip', type=float, default=0.1, help='PPO clip ratio')
    parser.add_argument('--T_train', type=int, default=200, help='number of itrations for training')
    parser.add_argument('--n_step', type=int, default=4, help='n_step for return estimation')
    parser.add_argument('--best_cl', action='store_true', help='use best solution found in CL as initial solution for training')
    parser.add_argument('--Xi_CL', type=float, default=0.25, help='hyperparameter of CL')
    parser.add_argument('--batch_size', type=int, default=600,help='number of instances per batch during training')
    parser.add_argument('--epoch_end', type=int, default=200, help='maximum training epoch')
    parser.add_argument('--epoch_size', type=int, default=12000, help='number of instances per epoch during training')
    parser.add_argument('--lr_model', type=float, default=1e-4, help="learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=3e-5, help="learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.985, help='learning rate decay per epoch')
    parser.add_argument('--max_grad_norm', type=float, default=0.04, help='maximum L2 norm for gradient clipping')
    
    # Inference and validation parameters
    parser.add_argument('--T_max', type=int, default=1500, help='number of steps for inference')
    parser.add_argument('--eval_only', action='store_true', help='switch to inference mode')
    parser.add_argument('--val_size', type=int, default=1000, help='number of instances for validation/inference')
    parser.add_argument('--val_dataset', type=str, default = './datasets/tsp_20_10000.pkl', help='dataset file path')
    parser.add_argument('--val_m', type=int, default=1, help='number of data augments (<=8)')

    # resume and load models
    parser.add_argument('--load_path', default = None, help='path to load model parameters and optimizer state from')
    parser.add_argument('--resume', default = None, help='resume from previous checkpoint file')
    parser.add_argument('--epoch_start', type=int, default=0, help='start at epoch # (relevant for learning rate decay)')

    # logs/output settings
    parser.add_argument('--no_progress_bar', action='store_true', help='disable progress bar')
    parser.add_argument('--log_dir', default='logs', help='directory to write TensorBoard information to')
    parser.add_argument('--log_step', type=int, default=50, help='log info every log_step gradient steps')
    parser.add_argument('--output_dir', default='outputs', help='directory to write output models to')
    parser.add_argument('--run_name', default='run_name', help='name to identify the run')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    
    opts = parser.parse_args(args)
    
    # figure out whether to use distributed training if needed
    opts.world_size = torch.cuda.device_count()
    opts.distributed = (torch.cuda.device_count() > 1) and (not opts.no_DDP)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'
    # processing settings
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.P = 250 if opts.eval_only else 1e10
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    ) if not opts.no_saving else None

    return opts