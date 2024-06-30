import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from models import model_utils
from torch.utils.checkpoint import checkpoint

class ActorStochasticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorStochasticMLP, self).__init__()

        self.device = device
        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))
            else:
                modules.append(model_utils.get_activation_func('identity'))

        self.mu_net = nn.Sequential(*modules).to(device)
        logstd = cfg_network.get('actor_logstd_init', -1.0)
        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)

    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic=False):
        def custom_forward(*inputs):
            return self.mu_net(inputs[0])

        mu = checkpoint(custom_forward, obs.requires_grad_())
        if deterministic:
            return mu
        else:
            std = self.logstd.exp()
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample

    def forward_with_dist(self, obs, deterministic=False):
        def custom_forward(*inputs):
            return self.mu_net(inputs[0])

        mu = checkpoint(custom_forward, obs.requires_grad_())
        std = self.logstd.exp()
        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std
        
    def evaluate_actions_log_probs(self, obs, actions):
        def custom_forward(*inputs):
            return self.mu_net(inputs[0])

        mu = checkpoint(custom_forward, obs.requires_grad_())
        std = self.logstd.exp()
        dist = Normal(mu, std)
        return dist.log_prob(actions)
