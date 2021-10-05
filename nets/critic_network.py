import torch
from torch import nn
from nets.graph_layers import  MultiHeadAttentionLayerforCritic, ValueDecoder


class Critic(nn.Module):

    def __init__(self,
             problem_name,
             embedding_dim,
             hidden_dim,
             n_heads,
             n_layers,
             normalization,
             ):
        
        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        
        self.encoder = nn.Sequential(*(
                MultiHeadAttentionLayerforCritic(self.n_heads, 
                                    self.embedding_dim * 2, 
                                    self.hidden_dim * 2, 
                                    self.normalization)
                        for _ in range(1)))
            
        self.value_head = ValueDecoder(input_dim = self.embedding_dim * 2,
                                       embed_dim = self.embedding_dim * 2)

        
    def forward(self, input):
        
        # get concatenated input
        h_features = torch.cat(input, -1).detach()
        
        # pass through encoder
        h_em = self.encoder(h_features)

        # pass through value_head, get estimated value
        baseline_value = self.value_head(h_em)
        
        return baseline_value.detach().squeeze(), baseline_value.squeeze()
        
