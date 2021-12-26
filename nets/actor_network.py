from torch import nn
import torch
from nets.graph_layers import MultiHeadEncoder, MultiHeadDecoder, EmbeddingNet
from torch.distributions import Categorical
import torch.nn.functional as F

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Actor(nn.Module):

    def __init__(self,
                 problem_name,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_heads_decoder,
                 n_layers,
                 normalization,
                 v_range,
                 seq_length,
                 ):
        super(Actor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_heads_decoder = n_heads_decoder        
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.seq_length = seq_length
        
        # see Appendix D
        if problem_name == 'tsp':
            self.node_dim = 2
        elif problem_name == 'cvrp':
            self.node_dim = 7
        else:            
            assert False, "Unsupported problem: {}".format(self.problem.NAME)        

        # build DACT model
        self.embedder = EmbeddingNet(
                            self.node_dim,
                            self.embedding_dim,
                            self.seq_length)
        
        self.encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor, 
                                self.embedding_dim, 
                                self.hidden_dim, 
                                self.normalization,)
            for _ in range(self.n_layers))) # stack L layers
            
        self.decoder = MultiHeadDecoder(n_heads = self.n_heads_decoder,
                                        input_dim = self.embedding_dim, 
                                        embed_dim = self.embedding_dim)
        
        print(self.get_parameter_number())

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, problem, x_in, solution, exchange, do_sample = False, fixed_action = None, require_entropy = False, to_critic = False, only_critic  = False):

        bs, gs, in_d = x_in.size()
        
        if problem.NAME == 'cvrp':
            
            # prepare the 7-dim features x_in
            argsort = solution.argsort()
            loc = x_in[:,:,:2]            
            post = loc.gather(1, solution.view(bs, -1, 1).expand_as(loc))
            pre = loc.gather(1, argsort.view(bs, -1, 1).expand_as(loc))
            
            post = torch.norm(post - loc, 2, -1, True)
            pre = torch.norm(pre - loc, 2, -1, True)
            
            
            x_in = torch.cat((loc, pre, post, x_in[:,:,-1:]), -1)
            del post, pre, argsort
            
            # get feasibility masks for current state
            mask_table, contex, to_actor = problem.get_swap_mask(solution, x_in)
            mask_table = mask_table.cpu()
            
            # concate the 7-dim features x_in
            x_in = torch.cat((x_in[:,:,:4], to_actor), -1)
            del to_actor
            contex = contex % 1000 // 1
            
            # pass through embedder to get embeddings
            NFE, PFE, visited_time = self.embedder(x_in, solution, visited_time = contex)
        
        elif problem.NAME == 'tsp':
            # pass through embedder
            NFE, PFE, visited_time = self.embedder(x_in, solution, None)
            
            # mask infeasible solutions
            mask_table = problem.get_swap_mask(visited_time).expand(bs, gs, gs).cpu()
        
        else: 
            raise NotImplementedError()
        
        del visited_time
        
        # pass through DAC encoder
        h_em, g_em = self.encoder(NFE, PFE)
        
        # share embeddings to critic net
        if only_critic:
            return (h_em, g_em)
        
        # pass through DAC decoder
        compatibility = torch.tanh(self.decoder(h_em, g_em, None)) * self.range

        # perform masking
        compatibility[mask_table] = -1e20
        
        del mask_table

        # mask the last action to aviod loops
        if exchange is not None:        
            compatibility[torch.arange(bs), exchange[:,0], exchange[:,1]] = -1e20
            compatibility[torch.arange(bs), exchange[:,1], exchange[:,0]] = -1e20            
        
        #reshape our tables
        im = compatibility.view(bs, -1)

        # softmax
        log_likelihood = F.log_softmax(im,dim = -1)
        M_table = F.softmax(im,dim = -1)

        # fixed action for PPO training if needed
        if fixed_action is not None:
            row_selected = fixed_action[:,0]
            col_selected = fixed_action[:,1]
            pair_index = row_selected * gs + col_selected
            pair_index = pair_index.view(-1,1)
            pair = fixed_action
            
        else:
            # sample one action
            if do_sample: 
                pair_index = M_table.multinomial(1)
            else:
                pair_index = M_table.max(-1)[1].view(-1,1)
        
            # from action (selected node pair)
            col_selected = pair_index % gs 
            row_selected = pair_index // gs 
            pair = torch.cat((row_selected, col_selected),-1)  # pair: no_head bs, 2
 
        selected_log_likelihood = log_likelihood.gather(1,pair_index)
    
        if require_entropy:
            
            dist = Categorical(M_table, validate_args=False) # for logging only
            entropy = dist.entropy() # for logging only
            
            out = (pair,
                   selected_log_likelihood.squeeze(),
                   (h_em, g_em) if to_critic else None,
                   entropy)
        else:
            out = (pair,
                   selected_log_likelihood.squeeze(),
                   (h_em, g_em) if to_critic else None)
            
        return out
