import math
import torch
import torch.nn as nn
import torch.distributions as D

class RepresentationLayer(torch.nn.Module):
    '''
    Implements a representation layer, that accumulates pytorch gradients.

    Representations are vectors in nrep-dimensional real space. By default
    they will be initialized as a tensor of dimension nsample x nrep from a
    normal distribution (mean and variance given by init).

    One can also supply a tensor to initialize the representations (values=tensor).
    The representations will then have the same dimension and will assumes that
    the first dimension is nsample (and the last is nrep).

    forward() takes a sample index and returns the representation.

    Representations are "linear", so a representation layer may be followed
    by an activation function.

    To update representations, the pytorch optimizers do not always work well,
    so the module comes with it's own SGD update (self.update(lr,mom,...)).

    If the loss has reduction="sum", things work well. If it is ="mean", the
    gradients become very small and the learning rate needs to be rescaled
    accordingly (batchsize*output_dim).

    Do not forget to call the zero_grad() before each epoch (not inside the loop
    like with the weights).

    '''
    def __init__(self,
                nrep,        # Dimension of representation
                nsample,     # Number of training samples
                init=(0.,1.),# Normal distribution mean and stddev for
                                # initializing representations
                values=None  # If values is given, the other parameters are ignored
                ):
        super(RepresentationLayer, self).__init__()
        self.dz = None
        if values is None:
            self.nrep=nrep
            self.nsample=nsample
            self.mean, self.stddev = init[0],init[1]
            self.init_random(self.mean,self.stddev)
        else:
            # Initialize representations from a tensor with values
            self.nrep = values.shape[-1]
            self.nsample = values.shape[0]
            self.mean, self.stddev = None, None
            # Is this the way to copy values to a parameter?
            self.z = torch.nn.Parameter(torch.zeros_like(values), requires_grad=True)
            with torch.no_grad():
                self.z.copy_(values)

    def init_random(self,mean,stddev):
        # Generate random representations
        self.z = torch.nn.Parameter(torch.normal(mean,stddev,size=(self.nsample,self.nrep), requires_grad=True))

    def forward(self, idx=None):
        if idx is None:
            return self.z
        else:
            return self.z[idx]

    # Index can be whatever it can be for a torch.tensor (e.g. tensor of idxs)
    def __getitem__(self,idx):
        return self.z[idx]

    def fix(self):
        self.z.requires_grad = False

    def unfix(self):
        self.z.requires_grad = True

    def zero_grad(self):  # Used only if the update function is used
        if self.z.grad is not None:
            self.z.grad.detach_()
            self.z.grad.zero_()

    def update(self,idx=None,lr=0.001,mom=0.9,wd=None):
        if self.dz is None:
            self.dz = torch.zeros(self.z.size()).to(self.z.device)
        with torch.no_grad():
            # Update z
            # dz(k,j) = sum_i grad(k,i) w(i,j) step(z(j))
            self.dz[idx] = self.dz[idx].mul(mom) - self.z.grad[idx].mul(lr)
            if wd is not None:
                self.dz[idx] -= wd*self.z[idx]
            self.z[idx] += self.dz[idx]

    def rescale(self):
        z_flat = torch.flatten(self.z.cpu().detach())
        sd, m = torch.std_mean(z_flat)
        with torch.no_grad():
            self.z -= m
            self.z /= sd