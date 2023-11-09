import math
import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
import json


class gaussian:
    """
    This is a simple Gaussian prior used for initializing mixture model means
    """

    def __init__(self, dim, mean, stddev):
        self.dim = dim
        self.mean = mean
        self.stddev = stddev
        self.g = torch.distributions.normal.Normal(mean, stddev)

    def sample(self, n):
        return self.g.sample((n, self.dim))

    def log_prob(self, x):
        return self.g.log_prob(x)


class softball:
    """
    Almost uniform prior for the m-dimensional ball.
    Logistic function makes a soft (differentiable) boundary.
    Returns a prior function and a sample function.
    The prior takes a tensor with a batch of z
    vectors (last dim) and returns a tensor of prior log-probabilities.
    The sample function returns n samples from the prior (approximate
    samples uniform from the m-ball). NOTE: APPROXIMATE SAMPLING.
    """

    def __init__(self, dim, radius, a=1):
        self.dim = dim
        self.radius = radius
        self.a = a
        self.norm = math.lgamma(1 + dim * 0.5) - dim * (
            math.log(radius) + 0.5 * math.log(math.pi)
        )

    def sample(self, n):
        # Return n random samples
        # Approximate: We sample uniformly from n-ball
        with torch.no_grad():
            # Gaussian sample
            sample = torch.randn((n, self.dim))
            # n random directions
            sample.div_(sample.norm(dim=-1, keepdim=True))
            # n random lengths
            local_len = self.radius * torch.pow(torch.rand((n, 1)), 1.0 / self.dim)
            sample.mul_(local_len.expand(-1, self.dim))
        return sample

    def log_prob(self, z):
        # Return log probabilities of elements of tensor (last dim assumed to be z vectors)
        return self.norm - torch.log(
            1 + torch.exp(self.a * (z.norm(dim=-1) / self.radius - 1))
        )


class GaussianMixture(nn.Module):
    def __init__(
        self,
        Nmix,
        dim,
        type="diagonal",
        alpha=1,
        mean_init=(1.0, 1.0),
        sd_init=[1.0, 1.0],
    ):
        """
        A mixture of multi-variate Gaussians

        Nmix is the number of components in the mixture
        dim is the dimension of the space
        type can be "fixed", "isotropic" or "diagonal", which refers to the covariance matrices
        mean_prior is a prior class with a log_prob and sample function
            - Standard normal if not specified.
            - Other option is ('softball',<radius>,<hardness>)
        If there is no mean_prior specified, a default Gaussian will be chosen with
            - mean_init[0] as mean and mean_init[1] as standard deviation
        logbeta_prior is a prior class for the negative log variance of the mixture components
            - logbeta = log (1/sigma^2)
            - If it is not specified, we make this prior a Gaussian from sd_init parameters
            - For the sake of interpretability, the sd_init parameters represent the desired mean and (approximately) sd of the standard deviation
            - the difference btw giving a prior beforehand and giving only init values is that with a given prior, the logbetas will be sampled from it, otherwise they will be initialized the same
        alpha determines the Dirichlet prior on mixture coefficients
        Mixture coefficients are initialized uniformly
        Other parameters are sampled from prior
        """
        super(GaussianMixture, self).__init__()
        self.dim = dim
        self.Nmix = Nmix
        # self.init = init

        # Means with shape: Nmix,dim
        self.mean = nn.Parameter(torch.empty(Nmix, dim), requires_grad=True)
        self.mean_prior = softball(self.dim, mean_init[0], mean_init[1])

        # Dirichlet prior on mixture
        self.alpha = alpha
        self.dirichlet_constant = math.lgamma(Nmix * alpha) - Nmix * math.lgamma(alpha)

        # Log inverse variance with shape (Nmix,dim) or (Nmix,1)
        self.sd_init = sd_init
        self.sd_init[0] = 0.2 * (mean_init[0] / self.Nmix)
        self.betafactor = dim * 0.5  # rename this!
        self.bdim = 1  # If 'diagonal' the dimension of lobbeta is = dim
        if type == "fixed":
            # No gradient needed for training
            # This is a column vector to be correctly broadcastet in std dev tensor
            self.logbeta = nn.Parameter(
                torch.empty(Nmix, self.bdim), requires_grad=False
            )
            self.logbeta_prior = None
        else:
            if type == "diagonal":
                self.betafactor = 0.5
                self.bdim = dim
            elif type != "isotropic":
                raise ValueError(
                    "type must be 'isotropic' (default), 'diagonal', or 'fixed'"
                )

            self.logbeta = nn.Parameter(
                torch.empty(Nmix, self.bdim), requires_grad=True
            )

        # Mixture coefficients. These are weights for softmax
        self.weight = nn.Parameter(torch.empty(Nmix), requires_grad=True)
        self.init_params()

        # -dim*0.5*log(2pi)
        self.pi_term = -0.5 * self.dim * math.log(2 * math.pi)

    def init_params(self):
        with torch.no_grad():
            # Means are sampled from the prior
            self.mean.copy_(self.mean_prior.sample(self.Nmix))
            self.logbeta.fill_(-2 * math.log(self.sd_init[0]))
            self.logbeta_prior = gaussian(
                self.bdim, -2 * math.log(self.sd_init[0]), self.sd_init[1]
            )

            # Weights are initialized to 1, corresponding to uniform mixture coeffs
            self.weight.fill_(1)

    def forward(self, x, label=None):
        # The beta values are obtained from logbeta
        halfbeta = 0.5 * torch.exp(self.logbeta)

        # y = logp =  - 0.5*log (2pi) -0.5*beta(x-mean[i])^2 + 0.5*log(beta)
        # sum terms for each component (sum is over last dimension)
        # y is one-dim with length Nmix
        # x is unsqueezed to (nsample,1,dim), so broadcasting of mean (Nmix,dim) works
        y = (
            self.pi_term
            - (x.unsqueeze(-2) - self.mean).square().mul(halfbeta).sum(-1)
            + self.betafactor * self.logbeta.sum(-1)
        )
        # For each component multiply by mixture probs
        y += torch.log_softmax(self.weight, dim=0)
        y = torch.logsumexp(y, dim=-1)
        y = y + self.prior()  # += gives cuda error

        return y

    def log_prob(self, x):  # Add label?
        self.forward(x)

    def mixture_probs(self):
        return torch.softmax(self.weight, dim=-1)

    def covariance(self):
        return torch.exp(-self.logbeta)

    def prior(self):
        """Calculate log prob of prior on mean, logbeta, and mixture coeff"""
        # Mixture
        p = self.dirichlet_constant  # /self.Nmix
        if self.alpha != 1:
            p = p + (self.alpha - 1.0) * (
                self.mixture_probs().log().sum()
            )  # /self.Nmix
        # Means
        p = p + self.mean_prior.log_prob(self.mean).sum()  # /self.Nmix
        # logbeta
        if self.logbeta_prior is not None:
            p = (
                p + self.logbeta_prior.log_prob(self.logbeta).sum()
            )  # /(self.Nmix*self.dim)
        return p

    def Distribution(self):
        with torch.no_grad():
            mix = D.Categorical(probs=torch.softmax(self.weight, dim=-1))
            comp = D.Independent(D.Normal(self.mean, torch.exp(-0.5 * self.logbeta)), 1)
            return D.MixtureSameFamily(mix, comp)

    def sample(self, nsample):
        with torch.no_grad():
            gmm = self.Distribution()
            return gmm.sample(torch.tensor([nsample]))

    def component_sample(self, nsample):
        """Returns a sample from each component. Tensor shape (nsample,nmix,dim)"""
        with torch.no_grad():
            comp = D.Independent(D.Normal(self.mean, torch.exp(-0.5 * self.logbeta)), 1)
            return comp.sample(torch.tensor([nsample]))

    def sample_probs(self, x):
        halfbeta = 0.5 * torch.exp(self.logbeta)
        y = (
            self.pi_term
            - (x.unsqueeze(-2) - self.mean).square().mul(halfbeta).sum(-1)
            + self.betafactor * self.logbeta.sum(-1)
        )
        y += torch.log_softmax(self.weight, dim=0)
        return torch.exp(y)

    """This section is for learning new data points"""

    def sample_new_points(self, n_points, option="random", n_new=1):
        """
        Generates samples for each new data point
            - n_points defines the number of new data points to learn
            - option defines which of 2 schemes to use
                - random: sample n_new vectors from each component
                    -> Nmix * n_new values per new point
                - mean: take the mean of each component as initial representation values
                    -> Nmix values per new point
        The order of repetition in both options is [a,a,a, b,b,b, c,c,c] on data point ID.
        """
        self.new_samples = n_new
        multiplier = self.Nmix
        if option == "random":
            out = self.component_sample(n_points * n_new)
            multiplier *= n_new

        elif option == "mean":
            with torch.no_grad():
                out = torch.repeat_interleave(
                    self.mean.clone().cpu().detach().unsqueeze(0), n_points, dim=0
                )
        else:
            print(
                "Please specify how to initialize new representations correctly \nThe options are 'random' and 'mean'."
            )
        return out.view(n_points * self.new_samples * self.Nmix, self.dim)

    def reshape_targets(self, y, y_type="true"):
        """
        Since we have multiple representations for the same new data point,
        we need to reshape the output a little to calculate the losses
        Depending on the y_type, y can be
            - the true targets (y_type: 'true')
            - the model predictions (y_type: 'predicted') (can also be used for rep.z in dataloader loop)
            - the 4-dimensional representation or the loss (y_type: 'reverse')
        """

        if y_type == "true":
            if len(y.shape) > 2:
                raise ValueError(
                    "Unexpected shape in input to function reshape_targets. Expected 2 dimensions, got "
                    + str(len(y.shape))
                )
            return (
                y.unsqueeze(1).unsqueeze(1).expand(-1, self.new_samples, self.Nmix, -1)
            )
        elif y_type == "predicted":
            if len(y.shape) > 2:
                raise ValueError(
                    "Unexpected shape in input to function reshape_targets. Expected 2 dimensions, got "
                    + str(len(y.shape))
                )
            n_points = int(
                torch.numel(y) / (self.new_samples * self.Nmix * y.shape[-1])
            )
            return y.view(n_points, self.new_samples, self.Nmix, y.shape[-1])
        elif "reverse":
            if len(y.shape) < 4:
                # this case is for when the losses are of shape (n_points,self.new_samples,self.Nmix)
                return y.view(y.shape[0] * self.new_samples * self.Nmix)
            else:
                return y.view(y.shape[0] * self.new_samples * self.Nmix, y.shape[-1])
        else:
            raise ValueError(
                "The y_type in function reshape_targets was incorrect. Please choose between 'true' and 'predicted'."
            )

    def choose_best_representations(self, x, losses):
        """
        Selects the representation for each new datapoint that maximizes the objective
            - x are the newly learned representations
            - x and losses have to have the same shape in the first dimension
              make sure that the losses are only summed over the output dimension
        Outputs new representation values
        """
        n_points = int(torch.numel(losses) / (self.new_samples * self.Nmix))

        best_sample = torch.argmin(
            losses.view(-1, self.new_samples * self.Nmix), dim=1
        ).squeeze(-1)
        best_rep = x.view(n_points, self.new_samples * self.Nmix, self.dim)[
            range(n_points), best_sample
        ]
        # best_rep = torch.diagonal(x.view(n_points,self.new_samples*self.Nmix,self.dim)[:,best_sample],dim1=0,dim2=1).transpose(0,1)

        return best_rep

    def choose_old_or_new(self, z_new, loss_new, z_old, loss_old):
        if (len(z_new.shape) == 2) and (len(z_old.shape) == 2):
            z_conc = torch.cat((z_new.unsqueeze(1), z_old.unsqueeze(1)), dim=1)
        else:
            raise ValueError(
                "Unexpected shape in input to function choose_old_or_new. Expected 2 dimensions for z_new and z_old, got "
                + str(len(z_new.shape))
                + " and "
                + str(len(z_old.shape))
            )

        len_loss_new = len(loss_new.shape)
        for l in range(3 - len_loss_new):
            loss_new = loss_new.unsqueeze(1)
        len_loss_old = len(loss_old.shape)
        for l in range(3 - len_loss_old):
            loss_old = loss_old.unsqueeze(1)
        losses = torch.cat((loss_new, loss_old), dim=1)

        best_sample = torch.argmin(losses, dim=1).squeeze(-1)
        # print(str(best_sample.sum().item())+' out of '+str(z_new.shape[0])+' samples were resampled.')

        best_rep = z_conc[range(z_conc.shape[0]), best_sample]

        return best_rep, round((best_sample.sum().item() / z_new.shape[0]) * 100, 2)

    def clustering(self, z):
        """compute the cluster assignment (as int) for each sample"""
        return torch.argmax(self.sample_probs(torch.tensor(z)), dim=-1).to(torch.int16)

    @classmethod
    def load(cls, save_dir="./"):
        # get saved hyper-parameters
        with open(save_dir + "dgd_hyperparameters.json", "r") as fp:
            param_dict = json.load(fp)

        gmm = cls(
            Nmix=param_dict["n_components"],
            dim=param_dict["latent"],
            type="diagonal",
            alpha=param_dict["dirichlet_a"],
            mean_init=(param_dict["mp_scale"], param_dict["hardness"]),
            sd_init=[param_dict["sd_mean"], param_dict["sd_sd"]],
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gmm.load_state_dict(
            torch.load(save_dir + param_dict["name"] + "_gmm.pt", map_location=device)
        )
        return gmm


class GaussianMixtureSupervised(GaussianMixture):
    def __init__(
        self,
        Nclass,
        dim,
        Ncpc=1,
        type="diagonal",
        alpha=1,
        mean_init=(1.0, 1.0),
        sd_init=(1.0, 1.0),
    ):
        super(GaussianMixtureSupervised, self).__init__(
            Ncpc * Nclass,
            dim,
            type=type,
            alpha=alpha,
            mean_init=mean_init,
            sd_init=sd_init,
        )
        self.dim = dim

        self.Nclass = Nclass
        self.Ncpc = Ncpc

    def forward(self, x, label=None):
        if label is None:
            y = super().forward(x)
            return y

        if 999 in label:
            # first get normal loss
            idx_unsup = [i for i in range(len(label)) if label[i] == 999]
            y_unsup = super().forward(x[idx_unsup])
            # Otherwise use the component corresponding to the label
            idx_sup = [i for i in range(len(label)) if label[i] != 999]
            halfbeta = 0.5 * torch.exp(self.logbeta)
            # Pick only the Nclc components belonging class
            # y_sup = self.pi_term - (x.unsqueeze(-2).unsqueeze(-2) - self.mean.view(self.Nclass,self.Ncpc,-1)).square().mul(halfbeta.unsqueeze(-2))[label[idx_sup]].sum(-1).sum(-1) + self.betafactor*self.logbeta.view(self.Nclass,self.Ncpc,-1).sum(-1).sum(-1)
            y_sup = (
                self.pi_term
                - (
                    x.unsqueeze(-2).unsqueeze(-2)
                    - self.mean.view(self.Nclass, self.Ncpc, -1)
                )
                .square()
                .mul(halfbeta.unsqueeze(-2))
                .sum(-1)
                + (self.betafactor * self.logbeta.view(self.Nclass, self.Ncpc, -1)).sum(
                    -1
                )
            )
            y_sup += torch.log_softmax(self.weight.view(self.Nclass, self.Ncpc), dim=-1)
            y_sup = y_sup.sum(-1)
            y_sup = torch.abs(
                y_sup[(idx_sup, label[idx_sup])] * self.Nclass
            )  # this is replacement for logsumexp of supervised samples
            # put together y
            y = torch.empty((x.shape[0]), dtype=torch.float32).to(x.device)
            y[idx_unsup] = y_unsup
            y[idx_sup] = y_sup
        else:
            halfbeta = 0.5 * torch.exp(self.logbeta)
            # Pick only the Nclc components belonging class
            y = (
                self.pi_term
                - (
                    x.unsqueeze(-2).unsqueeze(-2)
                    - self.mean.view(self.Nclass, self.Ncpc, -1)
                )
                .square()
                .mul(halfbeta.unsqueeze(-2))
                .sum(-1)
                + (self.betafactor * self.logbeta.view(self.Nclass, self.Ncpc, -1)).sum(
                    -1
                )
            )
            y += torch.log_softmax(self.weight.view(self.Nclass, self.Ncpc), dim=-1)
            y = y.sum(-1)
            # y = torch.abs(y[(np.arange(y.shape[0]),label)] * self.Nclass) # this is replacement for logsumexp of supervised samples
            y = (
                y[(np.arange(y.shape[0]), label)] * self.Nclass
            )  # this is replacement for logsumexp of supervised samples

        y = y + super().prior()
        return y

    def log_prob(self, x, label=None):
        self.forward(x, label=label)

    def label_mixture_probs(self, label):
        return torch.softmax(self.weight[label], dim=-1)
