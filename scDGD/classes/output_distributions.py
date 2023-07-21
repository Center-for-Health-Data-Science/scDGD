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

    The scaled value mhat is typically the output value from the NN, 
    but we are expanding to the option of modelling the probability.

    If rhat=None, it is assumed that the last half of mhat contains rhat.

    m = M*mhat
    '''
    def __init__(self, out_dim, r_init, scaling_type='max', output='mean', activation='sigmoid', reduction='none'):
        super(NBLayer, self).__init__()

        # initialize parameter for r
        # real-valued positive parameters are usually used as their log equivalent
        self.log_r = torch.nn.Parameter(torch.full(fill_value=math.log(r_init-1), size=(1,out_dim)), requires_grad=True)
        self.scaling_type = scaling_type
        self.output = output
        self.activation = activation
        self.reduction = reduction

        #self.activ_layer = nn.ModuleList()
        if self.activation == 'sigmoid':
            self.activ_layer = nn.Sigmoid()
        elif self.activation == 'softmax':
            self.activ_layer = nn.Softmax(dim=-1)
        elif self.activation == 'softplus':
            self.activ_layer = nn.Softplus()
        else:
            raise ValueError('Activation function not recognized')
    
    @property
    def r(self):
        return torch.exp(self.log_r)+1
    
    def forward(self, x):
        if self.output == 'mean':
            return self.activ_layer(x)
        elif self.output == 'prob':
            p = self.activ_layer(x)
            mean = self.r * (1-p)/p
            return mean

    # Convert to m from scaled variables
    def rescale(self,M,mhat):
        return M*mhat

    def loss(self,x,M,mhat,gene_id=None):
        # k has shape (nbatch,dim), M has shape (nbatch,1)
        # mhat has dim (nbatch,dim)
        # r has dim (1,dim)
        if gene_id is not None:
            loss = -logNBdensity(x,self.rescale(M,mhat), self.r[0,gene_id])
        else:
            loss = -logNBdensity(x,self.rescale(M,mhat), self.r)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()

    # The logprob of the tensor
    def logprob(self,x,M,mhat):
        return logNBdensity(x,self.rescale(M,mhat), self.r)

    def sample(self,nsample,M,mhat):
        # Note that torch.distributions.NegativeBinomial returns FLOAT and not int
        with torch.no_grad():
            m = self.rescale(M,mhat)
            #probs = m/(m+torch.exp(self.log_r))
            probs = self.r / (m + self.r)
            nb = torch.distributions.NegativeBinomial(self.r, probs=probs)
            return nb.sample([nsample]).squeeze()