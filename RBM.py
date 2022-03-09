import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units, k):
        '''
        Initialization of the RBM:\n
        ---> visible_units = number of units in visible layer\n
        ---> hidden_units = number of units in hidden layer\n
        ---> k = steps of contrastive divergence
        '''
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units))
        self.bias_v = nn.Parameter(torch.zeros(visible_units))
        self.bias_h = nn.Parameter(torch.zeros(hidden_units))
        self.k = k
    
    def sample_hidden(self, v):
        '''
        Input: visible units state v\n
        Return: p_h = prob. of h given v\n
        \t\tsample_h = the sampled hidden state 
        '''
        p_h = torch.sigmoid(F.linear(v, self.W, self.bias_h))
        sample_h = torch.bernoulli(p_h)
        return p_h, sample_h
    
    def sample_visible(self, h):
        '''
        Input: hidden units state v\n
        Return: p_v = prob. of v given h\n
        \t\tsample_v = the sampled visible state 
        '''
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.bias_v))
        sample_v = torch.bernoulli(p_v)
        return p_v, sample_v

    def forward(self, v):
        '''
        Forward method required by torch
        '''
        _, h_ = self.sample_hidden(v)

        for _ in range(self.k):
            p_v, v_ = self.sample_visible(h_)
            p_h, h_ = self.sample_hidden(v_)
        
        return v, v_
    
    def free_energy(self, v):
        '''
        Input: visible units state
        Return: free energy of RBM given that state
        '''
        vbias_term = v.mv(self.bias_v)
        weights_term = F.linear(v,self.W,self.bias_h)
        hidden_term = weights_term.exp().add(1).log().sum(1)
        return (-vbias_term-hidden_term).mean()
    
    def summary(self):
        '''
        Print info on parameters of the RBM
        '''
        print("Visible Units:", self.bias_v.size())
        print("Hidden Units:", self.bias_h.size())
        print("Total Number of Weights:", self.W.size())
        print("Steps of Contrastive Divergence:", self.k)