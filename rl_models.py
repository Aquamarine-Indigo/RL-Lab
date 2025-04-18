import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import wandb
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
	nn.init.orthogonal_(layer.weight, std)
	nn.init.constant_(layer.bias, bias_const)
	return layer

class ActorCritic_Discrete(nn.Module):
	def __init__(self, envs, actor_hidden_layers, critic_hidden_layers):
		super(ActorCritic_Discrete, self).__init__()
		actor_module_list = []
		critic_module_list = []
		for i in range(len(actor_hidden_layers)-1):
			actor_module_list.append(layer_init(nn.Linear(actor_hidden_layers[i], actor_hidden_layers[i+1])))
			actor_module_list.append(nn.Tanh())
		for i in range(len(critic_hidden_layers)-1):
			critic_module_list.append(layer_init(nn.Linear(critic_hidden_layers[i], critic_hidden_layers[i+1])))
			critic_module_list.append(nn.Tanh())
		self.actor = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), actor_hidden_layers[0])),
			nn.Tanh(),
			*actor_module_list,
			# nn.Tanh(),
			layer_init(nn.Linear(actor_hidden_layers[-1], envs.single_action_space.n), std=0.01),
		)
		self.critic = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), critic_hidden_layers[0])),
			nn.Tanh(),
			*critic_module_list,
			# nn.Tanh(),
			layer_init(nn.Linear(critic_hidden_layers[-1], 1), std=1.0)
		) 
	
	def get_value(self, x):
		return self.critic(x)
	
	def get_action_and_value(self, x, action=None):
		logits = self.actor(x)
		action_probs = Categorical(logits=logits)
		if action is None:
			action = action_probs.sample()
		return action, action_probs.log_prob(action), action_probs.entropy(), self.critic(x)
	

class ActorCritic_Continuous(nn.Module):
	def __init__(self, envs, actor_hidden_layers, critic_hidden_layers):
		super(ActorCritic_Continuous, self).__init__()
		actor_module_list = []
		critic_module_list = []
		for i in range(len(actor_hidden_layers)-1):
			actor_module_list.append(layer_init(nn.Linear(actor_hidden_layers[i], actor_hidden_layers[i+1])))
			actor_module_list.append(nn.Tanh())
		for i in range(len(critic_hidden_layers)-1):
			critic_module_list.append(layer_init(nn.Linear(critic_hidden_layers[i], critic_hidden_layers[i+1])))
			critic_module_list.append(nn.Tanh())
		self.actor_mean = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), actor_hidden_layers[0])),
			nn.Tanh(),
			*actor_module_list,
			# nn.Tanh(),
			layer_init(nn.Linear(actor_hidden_layers[-1], np.prod(envs.single_action_space.shape)), std=0.01),
		)
		self.critic = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), critic_hidden_layers[0])),
			nn.Tanh(),
			*critic_module_list,
			# nn.Tanh(),
			layer_init(nn.Linear(critic_hidden_layers[-1], 1), std=1.0)
		)
		self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

	def get_value(self, x):
		return self.critic(x)
	
	def get_action_and_value(self, x, action=None):
		action_mean = self.actor_mean(x)
		action_logstd = self.actor_logstd.expand_as(action_mean)
		action_std = torch.exp(action_logstd)
		probs = Normal(action_mean, action_std)
		if action is None:
			action = probs.sample()
		return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def make_agent(envs, layers=[[64, 64], [64, 64]], device='cpu', agent_type='actor-critic-continuous'):
	if agent_type == 'actor-critic-discrete':
		agent = ActorCritic_Discrete(envs, layers[0], layers[1]).to(device)
		print(agent)
		return agent
	elif agent_type == 'actor-critic-continuous':
		agent = ActorCritic_Continuous(envs, layers[0], layers[1]).to(device)
		print(agent)
		return agent
	else:
		raise NotImplementedError