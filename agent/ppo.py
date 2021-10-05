import os
import warnings
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger
import torch.multiprocessing as mp
import torch.distributed as dist

from utils import clip_grad_norms
from nets.actor_network import Actor
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model, move_to, move_to_cuda
from utils.logger import log_to_tb_train
from agent.utils import validate


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []    
        
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch

class PPO:
    def __init__(self, problem_name, size, opts):
        
        # figure out the options
        self.opts = opts
        
        # figure out the actor
        self.actor = Actor(
            problem_name = problem_name,
            embedding_dim = opts.embedding_dim,
            hidden_dim = opts.hidden_dim,
            n_heads_actor = opts.DACTencoder_head_num,
            n_heads_decoder = opts.DACTdecoder_head_num,
            n_layers = opts.n_encode_layers,
            normalization = opts.normalization,
            v_range = opts.v_range,
            seq_length = size + (1 if problem_name == 'pdp' else 0 )
        )
        
        if not opts.eval_only:
        
            # figure out the critic
            self.critic = Critic(
                    problem_name = problem_name,
                    embedding_dim = opts.embedding_dim,
                    hidden_dim = opts.hidden_dim,
                    n_heads = opts.critic_head_num,
                    n_layers = opts.n_encode_layers,
                    normalization = opts.normalization
                )
        
            # figure out the optimizer
            self.optimizer = torch.optim.Adam(
            [{'params': self.actor.parameters(), 'lr': opts.lr_model}] + 
            [{'params': self.critic.parameters(), 'lr': opts.lr_critic}])
            
            # figure out the lr schedule
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)
                
        
        print(f'Distributed: {opts.distributed}')
        
        if opts.use_cuda and not opts.distributed:
            
            self.actor.to(opts.device)
            if not opts.eval_only: self.critic.to(opts.device)
            
            if torch.cuda.device_count() > 1:
                self.actor = torch.nn.DataParallel(self.actor)
                if not opts.eval_only: self.critic = torch.nn.DataParallel(self.critic)
                
    
    def load(self, load_path):
        
        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        
        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})
        
        if not self.opts.eval_only:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))
        
    
    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    
    
    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        if not self.opts.eval_only: self.critic.eval()
        
    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        if not self.opts.eval_only: self.critic.train()
    
    def rollout(self, problem, val_m, batch, do_sample = False, record = False, record_best = True, show_bar = False):
        
        # get instances and do data augments
        assert val_m <= 8
        batch = move_to(batch, self.opts.device)
        bs, gs, dim = batch['coordinates'].size()
        batch['coordinates'] = batch['coordinates'].unsqueeze(1).repeat(1,val_m,1,1)      
        if problem.NAME == 'cvrp': batch['demand'] = batch['demand'].unsqueeze(1).repeat(1,val_m,1).view(-1, gs)   
        
        for i in range(val_m):
            if i == 1: 
                batch['coordinates'][:,i,0] = 1 - batch['coordinates'][:,i,0]
            elif i==2:
                batch['coordinates'][:,i,1] = 1 - batch['coordinates'][:,i,1]
            elif i==3:
                batch['coordinates'][:,i,0] = 1 - batch['coordinates'][:,i,0]
                batch['coordinates'][:,i,1] = 1 - batch['coordinates'][:,i,1]
            elif i==4:
                batch['coordinates'][:,i,0] = batch['coordinates'][:,0,1]
                batch['coordinates'][:,i,1] = batch['coordinates'][:,0,0]
            elif i==5:
                batch['coordinates'][:,i,0] = 1 - batch['coordinates'][:,0,1]
                batch['coordinates'][:,i,1] = batch['coordinates'][:,0,0]
            elif i==6:
                batch['coordinates'][:,i,0] = batch['coordinates'][:,0,1]
                batch['coordinates'][:,i,1] = 1 - batch['coordinates'][:,0,0]
            elif i==7:
                batch['coordinates'][:,i,0] = 1 - batch['coordinates'][:,0,1]
                batch['coordinates'][:,i,1] = 1 - batch['coordinates'][:,0,0]
 
        batch['coordinates'] =  batch['coordinates'].view(-1, gs, dim)
        
        # get initial solutions
        solutions = move_to(problem.get_initial_solutions(batch), self.opts.device).long()
        # get initial cost
        obj = problem.get_costs(batch, solutions)
        
        obj_history = [torch.cat((obj[:,None],obj[:,None]),-1)]
        reward = []
        solution_history = [solutions.clone()]
        if record_best: best_solution = solutions.clone()
        else: best_solution = None
        
        # prepare the features
        batch_feature = problem.input_feature_encoding(batch)
        
        action = None
        solving_state = torch.zeros((batch_feature.size(0),1), device = self.opts.device).long()
        
        for t in tqdm(range(self.opts.T_max), disable = self.opts.no_progress_bar or not show_bar, 
                      desc = 'rollout', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            
             # pass through model
            action = self.actor(problem,
                                  batch_feature,
                                  solutions,
                                  action,
                                  do_sample = do_sample)[0]

            # state trasition
            solutions, rewards, obj, solving_state = problem.step(batch, 
                                                                  solutions, 
                                                                  action, 
                                                                  obj, 
                                                                  solving_state, 
                                                                  best_solution = best_solution)

            # record informations
            if record_best: best_solution[rewards > 0] = solutions[rewards > 0]
            reward.append(rewards)  
            obj_history.append(obj)
            if record: solution_history.append(solutions)
            
        
        out = (obj[:,-1].reshape(bs, val_m).min(1)[0], # batch_size, 1
               torch.stack(obj_history,1)[:,:,0].view(bs, val_m, -1).min(1)[0],  # batch_size, T
               torch.stack(obj_history,1)[:,:,-1].view(bs, val_m, -1).min(1)[0],  # batch_size, T
               torch.stack(reward,1).view(bs, val_m, -1).max(1)[0], # batch_size, T
               None if not record else torch.stack(solution_history,1))
        
        return out
      
    def start_inference(self, problem, val_dataset, tb_logger):
        if self.opts.distributed:            
            mp.spawn(validate, nprocs=self.opts.world_size, args=(problem, self, val_dataset, tb_logger, True))
        else:
            validate(0, problem, self, val_dataset, tb_logger, distributed = False)
            
    def start_training(self, problem, val_dataset, tb_logger):
        if self.opts.distributed:
            mp.spawn(train, nprocs=self.opts.world_size, args=(problem, self, val_dataset, tb_logger))
        else:
            train(0, problem, self, val_dataset, tb_logger)
      
def train(rank, problem, agent, val_dataset, tb_logger):
    
    opts = agent.opts     
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
        
    if opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=opts.world_size, rank = rank)
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.critic.to(device)
        for state in agent.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        
        if torch.cuda.device_count() > 1:
            agent.actor = torch.nn.parallel.DistributedDataParallel(agent.actor,
                                                                   device_ids=[rank])
            if not opts.eval_only: agent.critic = torch.nn.parallel.DistributedDataParallel(agent.critic,
                                                                   device_ids=[rank])
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
    else:
        for state in agent.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)
                        
    if opts.distributed: dist.barrier()
    
    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        
        # Training mode
        agent.lr_scheduler.step(epoch)
        if rank == 0:
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'], 
                                                                                 agent.optimizer.param_groups[1]['lr'], opts.run_name) , flush=True)
        # prepare training data
        training_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, DUMMY_RATE = opts.dummy_rate)
        if opts.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=False)
            training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size // opts.world_size, shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=train_sampler)
        else:
            training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, shuffle=False,
                                                       num_workers=0,
                                                       pin_memory=True)
            
        # start training
        step = epoch * (opts.epoch_size // opts.batch_size)  
        pbar = tqdm(total = (opts.K_epochs) * (opts.epoch_size // opts.batch_size) * (opts.T_train // opts.n_step) ,
                    disable = opts.no_progress_bar or rank!=0, desc = 'training',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for batch_id, batch in enumerate(training_dataloader):
            train_batch(rank,
                        problem,
                        agent,
                        epoch,
                        step,
                        batch,
                        tb_logger,
                        opts,
                        pbar)
            step += 1
        pbar.close()
        
        # save new model after one epoch  
        if rank == 0 and not opts.distributed: 
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
        elif opts.distributed and rank == 1:
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
            
        
        # validate the new model        
        if rank == 0 and not opts.distributed: validate(rank, problem, agent, val_dataset, tb_logger, _id = epoch)
        if rank == 0 and opts.distributed: validate(rank, problem, agent, val_dataset, tb_logger, _id = epoch)
        
        # syn
        if opts.distributed: dist.barrier()

    
def train_batch(
        rank,
        problem,
        agent,
        epoch,
        step,
        batch,
        tb_logger,
        opts,
        pbar):

    # setup
    agent.train()
    problem.train()
    memory = Memory()

    # prepare the input
    batch = move_to_cuda(batch, rank) if opts.distributed else move_to(batch, opts.device)
    batch_feature = problem.input_feature_encoding(batch).cuda() if opts.distributed \
                        else move_to(problem.input_feature_encoding(batch), opts.device)
    batch_size = batch_feature.size(0)
    action = move_to_cuda(torch.tensor([1,1]).repeat(batch_size,1), rank) if opts.distributed \
                        else move_to(torch.tensor([1,1]).repeat(batch_size,1), opts.device)
    solving_state = move_to_cuda(torch.zeros((batch_feature.size(0),1)).long(), rank) if opts.distributed \
                        else move_to(torch.zeros((batch_feature.size(0),1)).long(), opts.device)

    # initial solution
    solution = move_to_cuda(problem.get_initial_solutions(batch),rank) if opts.distributed \
                        else move_to(problem.get_initial_solutions(batch), opts.device)
    obj = problem.get_costs(batch, solution)
    
    # CL strategy
    if opts.Xi_CL:
        agent.eval()
        problem.eval(perturb = False)
        
        if opts.best_cl:
            solution_best = solution.clone()
        
        x = np.linspace(0,opts.epoch_end-1,opts.epoch_end)
        k = 0.02
        y = (1) / (1 + np.exp( (-k * x) + opts.epoch_end / 2 * k ))
        y = (y - y.min()) / (y.max() - y.min())
        z = y * opts.epoch_end * opts.Xi_CL
        
        for w in range(int(z[epoch])):
            
            # get model output	
            action = agent.actor( problem,
                                    batch_feature,
                                    solution,
                                    action,
                                    do_sample = True)[0]
             
            # state transient	
            solution, rewards, obj, solving_state = problem.step(batch, solution, action, obj, solving_state)
            
            if opts.best_cl:
                index = obj[:,0] == obj[:,1]
                solution_best[index] = solution[index]
        
        if opts.best_cl:
            solution = solution_best
            
        obj = problem.get_costs(batch, solution)
        
        agent.train()
        problem.train()    
    
    # params for training
    gamma = opts.gamma
    n_step = opts.n_step
    T = opts.T_train
    K_epochs = opts.K_epochs
    eps_clip = opts.eps_clip
    t = 0
    initial_cost = obj
    solving_state = torch.zeros((batch_feature.size(0),1)).long().cuda() if opts.distributed \
                    else torch.zeros((batch_feature.size(0),1), device = opts.device).long()
    
    # sample trajectory
    while t < T:
        t_s = t
        total_cost = 0
        entropy = []
        bl_val_detached = []
        bl_val = []
        memory.actions.append(action)
        
        while t - t_s < n_step and not (t == T):          
            
            
            memory.states.append(solution)
            
            # get model output
            action, log_lh, _to_critic, entro_p  = agent.actor(problem,
                                                               batch_feature,
                                                               solution,
                                                               action,
                                                               do_sample = True,
                                                               require_entropy = True,
                                                               to_critic = True)
            
            memory.actions.append(action)
            memory.logprobs.append(log_lh)
            
                            
            entropy.append(entro_p.detach().cpu())
            
            baseline_val_detached, baseline_val = agent.critic(_to_critic)
            
            bl_val_detached.append(baseline_val_detached)
            bl_val.append(baseline_val)
                
            # state transient
            solution, rewards, obj, solving_state = problem.step(batch, solution, action, obj, solving_state)
            memory.rewards.append(rewards)
            
            # store info
            total_cost = total_cost + obj[:,-1]
            
            # next            
            t = t + 1
            
            
        # store info
        t_time = t - t_s
        total_cost = total_cost / t_time
        
        # begin update        ======================= 
        
        # convert list to tensor
        all_actions = torch.stack(memory.actions)
        old_states = torch.stack(memory.states).detach().view(t_time, batch_size, -1)
        old_actions = all_actions[1:].view(t_time, -1, 2)
        old_logprobs = torch.stack(memory.logprobs).detach().view(-1)
        old_exchange = all_actions[:-1].view(t_time, -1, 2)
            
        # Optimize PPO policy for K mini-epochs:
        old_value = None
        for _k in range(K_epochs):
            
            if _k == 0:
                logprobs = memory.logprobs
                
            else:
                # Evaluating old actions and values :
                logprobs = []  
                entropy = []
                bl_val_detached = []
                bl_val = []
                
                for tt in range(t_time):    
                
                    # get new action_prob
                    _, log_p, _to_critic, entro_p = agent.actor(problem,
                                                                batch_feature,
                                                                old_states[tt],
                                                                old_exchange[tt],
                                                                # memory.solving_state[tt].float() / T,
                                                                fixed_action = old_actions[tt],
                                                                require_entropy = True,# take same action
                                                                to_critic = True)
                    
                    logprobs.append(log_p)
                    entropy.append(entro_p.detach().cpu())
                    
                    baseline_val_detached, baseline_val = agent.critic(_to_critic)
                    
                    bl_val_detached.append(baseline_val_detached)
                    bl_val.append(baseline_val)
            
            logprobs = torch.stack(logprobs).view(-1)
            entropy = torch.stack(entropy).view(-1)
            bl_val_detached = torch.stack(bl_val_detached).view(-1)
            bl_val = torch.stack(bl_val).view(-1)


            # get traget value for critic
            Reward = []
            reward_reversed = memory.rewards[::-1]
            
            # get next value
            R = agent.critic(agent.actor(problem,batch_feature,solution,action,only_critic = True))[0]
            for r in range(len(reward_reversed)):
                R = R * gamma + reward_reversed[r]
                Reward.append(R)
            
            # clip the target:
            Reward = torch.stack(Reward[::-1], 0)
            Reward = Reward.view(-1)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss:
            advantages = Reward - bl_val_detached

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages            
            reinforce_loss = -torch.min(surr1, surr2).mean()
            
            # define baseline loss
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                baseline_loss = v_max.mean()
            
            # check K-L divergence (for logging only)
            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
            
            # calculate loss      
            loss = baseline_loss + reinforce_loss
            
            # update gradient step
            agent.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradient norm and get (clipped) gradient norms for logging
            current_step = int(step * T / n_step * K_epochs + t//n_step * K_epochs  + _k)
            grad_norms = clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)
            
            # perform gradient descent
            agent.optimizer.step()
    
            # Logging to tensorboard            
            if(not opts.no_tb) and rank == 0:
                if current_step % int(opts.log_step) == 0:
                    log_to_tb_train(tb_logger, agent, Reward, ratios, bl_val_detached, total_cost, grad_norms, memory.rewards, entropy, approx_kl_divergence,
                       reinforce_loss, baseline_loss, logprobs, initial_cost, opts.show_figs, current_step)
                    
            if rank == 0: pbar.update(1)     
        
        # end update
        memory.clear_memory()