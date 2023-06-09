import math
import torch
import torch.nn as nn
import torch.distributions as D

def logNBdensity(k,m,r):
    ''' 
    Negative Binomial NB(k;m,r), where m is the mean and r is "number of failures"
    r can be real number (and so can k)
    k, and m are tensors of same shape
    r is tensor of shape (1, n_genes)
    Returns the log NB in same shape as k
    '''
    # remember that gamma(n+1)=n!
    eps = 1.e-10
    x = torch.lgamma(k+r)
    x -= torch.lgamma(r)
    x -= torch.lgamma(k+1)
    x += k*torch.log(m*(r+m+eps)**(-1)+eps)
    x += r*torch.log(r*(r+m+eps)**(-1))
    return x

class NBLayer(nn.Module):
    '''
    A negative binomial of scaled values of m and learned parameters for r.
    mhat = m/M, where M is the scaling factor

    The scaled value mhat is typically the output value from the NN

    If rhat=None, it is assumed that the last half of mhat contains rhat.

    m = M*mhat
    '''
    def __init__(self, out_dim, r_init, scaling_type='max', reduction='none'):
        super(NBLayer, self).__init__()

        # initialize parameter for r
        # real-valued positive parameters are usually used as their log equivalent
        self.log_r = torch.nn.Parameter(torch.full(fill_value=math.log(r_init-1), size=(1,out_dim)), requires_grad=True)
        # determine the type of activation based on scaling type
        if scaling_type == 'max':
            self.activation = 'sigmoid'
        elif scaling_type == 'total_count':
            self.activation = 'softmax'
        elif scaling_type in ['mean','median']:
            self.activation = 'softplus'
        else:
            raise ValueError('Unknown scaling type specified. Please use one of: "library", "total_count", "mean", or "median".')
        self.reduction = reduction
    
    def forward(self, x):
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'softmax':
            return torch.softmax(x, dim=-1)
        else:
            return F.softplus(x)

    # Convert to m from scaled variables
    def rescale(self,M,mhat):
        return M*mhat

    def loss(self,x,M,mhat,gene_id=None):
        # k has shape (nbatch,dim), M has shape (nbatch,1)
        # mhat has dim (nbatch,dim)
        # r has dim (1,dim)
        if gene_id is not None:
            loss = -logNBdensity(x,self.rescale(M,mhat),(torch.exp(self.log_r)+1)[0,gene_id])
        else:
            loss = -logNBdensity(x,self.rescale(M,mhat),(torch.exp(self.log_r)+1))
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()

    # The logprob of the tensor
    def logprob(self,x,M,mhat):
        return logNBdensity(x,self.rescale(M,mhat),(torch.exp(self.log_r)+1))

    def sample(self,nsample,M,mhat):
        # Note that torch.distributions.NegativeBinomial returns FLOAT and not int
        with torch.no_grad():
            m = self.rescale(M,mhat)
            probs = m/(m+torch.exp(self.log_r))
            nb = torch.distributions.NegativeBinomial((torch.exp(self.log_r)+1), probs=probs)
            return nb.sample([nsample]).squeeze()