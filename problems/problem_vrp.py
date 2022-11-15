from torch.utils.data import Dataset
import torch
import pickle
import os
import numpy as np


CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.,
            # add new data if needed
        }


class CVRP(object):

    NAME = 'cvrp'  # Capacitiated Vehicle Routing Problem
    
    def __init__(self, p_size, init_val_met = 'greedy', with_assert = False, step_method = '2_opt', P = 250, DUMMY_RATE = 0.2):
        
        self.size = int(np.ceil(p_size * (1 + DUMMY_RATE)))   # the number of real nodes plus dummy nodes in cvrp
        self.real_size = p_size # the number of real nodes in cvrp
        self.dummy_size = self.size - self.real_size
        self.do_assert = with_assert
        self.step_method = step_method
        self.init_val_met = init_val_met
        self.P = P
        print(f'CVRP with {self.real_size} nodes and {self.dummy_size} dummy depot.\n', 
              ' Do assert:', with_assert)
        self.train()
    
    def eval(self, perturb = True):
        self.training = False
        self.do_perturb = perturb
        
    def train(self):
        self.training = True
        self.do_perturb = False
    
    def input_feature_encoding(self, batch):
        return torch.cat((batch['coordinates'], batch['demand'].unsqueeze(-1)), -1) # solution-independent features

    def get_real_mask(self, rec, batch):
        
        # get mixed contex: 1000 * route_plan + 1 * visited_time + 0.5 * cu_demand  
        # (e.g., 1000 + 34 + 0.05 means the node is the 34th node in route 1 and the cum demand before the node is 0.1)
        contex, patial_sum = self.preprocessing(rec, batch)
        
        if self.step_method == '2_opt':
            
            # only allow in-route 2-opt
            route_plan = (contex // 1000).long() % self.dummy_size     
            mask_in = route_plan.view(-1,self.size,1) != route_plan.view(-1,1,self.size)
            mask_in[:,:self.dummy_size,:] = True
            mask_in[:,:, :self.dummy_size] = True
            # special case
            mask_special1 = mask_in.clone() & False
            mask_special1[:,self.dummy_size:,:] = True
            mask_special1 |= ((route_plan.view(-1,self.size,1) - 1) % self.dummy_size) != route_plan.view(-1,1,self.size)
            mask_special2 = mask_in.clone() & False
            mask_special2[:,:,self.dummy_size:] = True
            mask_special2 |= ((route_plan.view(-1,self.size,1)) % self.dummy_size) != route_plan.view(-1,1,self.size)
            
            # further allow btw-route 2-opt
            demand = batch['demand'] if isinstance(batch, dict) else batch[:,:,-1]
            cum_demand = ((contex % 1) * 2)
            total = patial_sum.gather(-1, route_plan)
            cor = (demand != 0).float() 
            pi = cum_demand.view(-1,self.size,1)
            pj = (cor * cum_demand).view(-1,1,self.size) 
            qi = (cor * (total - cum_demand)).view(-1,self.size,1)
            qj = (total - cor * cum_demand).view(-1,1,self.size)
            corj = demand.view(-1,1,self.size)
            mask_btw = ((pi + pj + corj) > (1.+0.1/CAPACITIES[self.real_size])) | ((qi + qj - corj) > (1.+0.1/CAPACITIES[self.real_size]))
            mask =  ~(~mask_in + ~mask_btw + ~mask_special1 + ~mask_special2)
            mask[:,:self.dummy_size, :self.dummy_size] = False

            return mask, contex, torch.cat((cum_demand.view(-1, self.size, 1),
                                            demand.view(-1, self.size, 1),
                                            (total - cor.view(-1,self.size) * cum_demand).view(-1, self.size, 1),
                                            ), -1
                                            ) 
        else:
            raise NotImplementedError()
        
    
    def get_initial_solutions(self, batch):
        
        batch_size = batch['coordinates'].size(0)
    
        def get_solution(methods):
            p_size = self.size
            
            if methods == 'random':
                
                candidates = torch.ones(batch_size,self.size).bool()
                candidates[:,:self.dummy_size] = False
                
                rec = torch.zeros(batch_size, self.size).long()
                selected_node = torch.zeros(batch_size, 1).long()
                cum_demand = torch.zeros(batch_size, 2)
                
                demand = batch['demand'].cpu()
                
                for i in range(self.size - 1):
                    
                    dists = torch.arange(p_size).view(-1, p_size).repeat(batch_size, 1)
                    
                    dists.scatter_(1, selected_node, 1e5)
                    dists[~candidates] = 1e5
                    
                    dists[cum_demand[:,-1:] + demand > 1.] = 1e5
                    dists.scatter_(1,cum_demand[:,:-1].long() + 1, 1e4)
                    
                    next_selected_node = dists.min(-1)[1].view(-1,1)
                    selected_demand = demand.gather(1,next_selected_node)
                    cum_demand[:,-1:] = torch.where(selected_demand >0, selected_demand + cum_demand[:,-1:], 0 * cum_demand[:,-1:])
                    cum_demand[:,:-1] = torch.where(selected_demand >0, cum_demand[:,:-1], cum_demand[:,:-1] + 1)
      
                    
                    rec.scatter_(1,selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node   
                    
                return rec
            
            
            elif methods == 'greedy':

                candidates = torch.ones(batch_size,self.size).bool()
                candidates[:,:self.dummy_size] = False
                
                rec = torch.zeros(batch_size, self.size).long()
                selected_node = torch.zeros(batch_size, 1).long()
                cum_demand = torch.zeros(batch_size, 2)
                
                d2 = batch['coordinates'].cpu()
                demand = batch['demand'].cpu()
                
                for i in range(self.size - 1):
                    
                    d1 = batch['coordinates'].cpu().gather(1, selected_node.unsqueeze(-1).expand(batch_size, self.size, 2))
                    dists = (d1 - d2).norm(p=2, dim=2)
                    
                    dists.scatter_(1, selected_node, 1e5)
                    dists[~candidates] = 1e5
                    
                    dists[cum_demand[:,-1:] + demand > 1.] = 1e5
                    dists.scatter_(1,cum_demand[:,:-1].long() + 1, 1e4)
                    
                    next_selected_node = dists.min(-1)[1].view(-1,1)
                    selected_demand = demand.gather(1,next_selected_node)
                    cum_demand[:,-1:] = torch.where(selected_demand >0, selected_demand + cum_demand[:,-1:], 0 * cum_demand[:,-1:])
                    cum_demand[:,:-1] = torch.where(selected_demand >0, cum_demand[:,:-1], cum_demand[:,:-1] + 1)
      
                    
                    rec.scatter_(1,selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node                          

                return rec
            
            else:
                raise NotImplementedError()

        return get_solution(self.init_val_met).expand(batch_size, self.size).clone()
    
    def step(self, batch, rec, exchange, pre_bsf, solving_state = None, best_solution = None):

        bs = exchange.size(0)
        pre_bsf = pre_bsf.view(bs,-1)
        
        first = exchange[:,0].view(bs,1)
        second = exchange[:,1].view(bs,1)
        
        if self.step_method  == 'swap':
            next_state = self.swap(rec, first, second)
        elif self.step_method  == '2_opt':            
            next_state = self.two_opt(rec, first, second)
        elif self.step_method  == 'insert':
            next_state = self.insert(rec, first, second)
        else:
            raise NotImplementedError()
        
        new_obj = self.get_costs(batch, next_state)
        
        now_bsf = torch.min(torch.cat((new_obj[:,None], pre_bsf[:,-1, None]),-1),-1)[0]
        
        reward = pre_bsf[:,-1] - now_bsf
        
        # update solving state
        solving_state[:,:1] = (1 - (reward > 0).view(-1,1).long()) * (solving_state[:,:1] + 1)
        
        
        if self.do_perturb:
            
            perturb_index = (solving_state[:,:1] > self.P).view(-1)
            solving_state[:,:1][perturb_index.view(-1, 1)] *= 0
            pertrb_cnt = perturb_index.sum().item()
            
            if pertrb_cnt > 0:
                next_state[perturb_index] =  best_solution[perturb_index]
            
            return next_state, reward, torch.cat((new_obj[:,None], now_bsf[:,None]),-1) , solving_state
        
        return next_state, reward, torch.cat((new_obj[:,None], now_bsf[:,None]),-1) , solving_state
    
    def two_opt(self, solution, first, second, is_perturb = False):
        
        rec = solution.clone()
        
        # fix connection for first node
        argsort = solution.argsort()
        pre_first = argsort.gather(1,first)  
        pre_first = torch.where(pre_first != second, pre_first, first)
        rec.scatter_(1,pre_first,second)
        
        # fix connection for second node
        post_second = solution.gather(1,second)
        post_second = torch.where(post_second != first, post_second, second)
        rec.scatter_(1,first, post_second)
        
        # reverse loop:
        cur = first
        for i in range(self.size):
            cur_next = solution.gather(1,cur)
            rec.scatter_(1,cur_next, torch.where(cur != second,cur,rec.gather(1,cur_next)))
            cur = torch.where(cur != second, cur_next, cur)
        
        return rec
       
    def check_feasibility(self, rec, batch):
        
        p_size = self.size

        assert (
            torch.arange(p_size).to(rec.device).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ).all(), "not visiting all nodes"
        
        real_rec = get_real_seq(rec)
        
        assert (
            torch.arange(p_size).to(rec.device).view(1, -1).expand_as(real_rec)  == 
            real_rec.sort(1)[0]
        ).all(), "not visiting all nodes"
        
        partial_sum = self.preprocessing(rec, batch)[-1]
        
        assert (partial_sum <= 1 + 1e-5).all(), ("not satisfying capacity constraint")
    
    
    def get_swap_mask(self, rec, batch):
        
        bs, gs = rec.size()        
        selfmask = torch.eye(gs, device = rec.device).view(1,gs,gs)
        
        real_mask, contex, to_actor = self.get_real_mask(rec, batch)
        masks = (real_mask) + selfmask.expand(bs,gs,gs).bool()
        
        return masks, contex, to_actor
        
    def get_costs(self, batch, rec):
        
        batch_size, size = rec.size()
        
        # check feasibility
        if self.do_assert:
            self.check_feasibility(rec, batch)
        
        # calculate obj value
        d1 = batch['coordinates'].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
        d2 = batch['coordinates']
        length =  ((d1  - d2).norm(p=2, dim=2)).sum(1)

        return length
    
    def preprocessing(self, solutions, batch):
        
        batch_size, seq_length = solutions.size()
        demand = batch['demand'] if isinstance(batch, dict) else batch[:,:,-1]
        arange = torch.arange(batch_size)
        
        pre = torch.zeros(batch_size, device = solutions.device).long()
        route = torch.zeros(batch_size, device = solutions.device)
        partial_sum = torch.zeros((batch_size, self.dummy_size), device = solutions.device)
        route_plan1000_visited_time1_dot_demand = torch.zeros((batch_size,seq_length), device = solutions.device)
        assert seq_length < 1000
        for i in range(seq_length):
            next_ = solutions[arange,pre]
            index = next_ < self.dummy_size
            route = torch.where(index, route + 1, route)
            cu_demand = torch.where(index, 
                                    partial_sum[arange, (route.long() - 1) % self.dummy_size], 
                                    partial_sum[arange, route.long() % self.dummy_size])
            route_plan1000_visited_time1_dot_demand[arange,next_] = i+1 + route * 1000 + cu_demand * 0.5
            partial_sum[arange,route.long() % self.dummy_size] += demand[arange, next_]
            pre = next_
            if self.do_assert: assert (cu_demand <=1+1e-5).all()
            
        return route_plan1000_visited_time1_dot_demand, partial_sum
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return CVRPDataset(*args, **kwargs)


class CVRPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0, distribution=None, DUMMY_RATE = None):
        
        super(CVRPDataset, self).__init__()
        
        self.data = []
        self.size = int(np.ceil(size * (1 + DUMMY_RATE))) # the number of real nodes plus dummy nodes in cvrp
        self.real_size = size # the number of real nodes in cvrp

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [self.make_instance(args) for args in data[offset:offset+num_samples]]

        else:            
            self.data = [{'coordinates': torch.cat((torch.FloatTensor(1, 2).uniform_(0, 1).repeat(self.size - self.real_size,1), 
                                                    torch.FloatTensor(self.real_size, 2).uniform_(0, 1)), 0),
                          'demand': torch.cat((torch.zeros(self.size - self.real_size),
                                               torch.FloatTensor(self.real_size).uniform_(1, 10).long() / CAPACITIES[self.real_size]), 0)
                          } for i in range(num_samples)]
            
            
        
        self.N = len(self.data)
        print(f'{self.N} instances initialized.')
    
    def make_instance(self, args):
        depot, loc, demand, capacity, *args = args
        
        depot = torch.FloatTensor(depot)
        loc = torch.FloatTensor(loc)
        demand = torch.FloatTensor(demand)
        
        return {'coordinates': torch.cat((depot.view(-1, 2).repeat(self.size - self.real_size,1), loc), 0),
                'demand': torch.cat((torch.zeros(self.size - self.real_size), demand / capacity), 0) }
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]


def get_real_seq(solutions):
    batch_size, seq_length = solutions.size()
    visited_time = torch.zeros((batch_size,seq_length)).to(solutions.device)
    pre = torch.zeros((batch_size),device = solutions.device).long()
    for i in range(seq_length):
       visited_time[torch.arange(batch_size),solutions[torch.arange(batch_size),pre]] = i+1
       pre = solutions[torch.arange(batch_size),pre]
       
    visited_time = visited_time % seq_length
    return visited_time.argsort()  
