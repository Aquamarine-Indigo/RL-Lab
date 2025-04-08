import argparse
import os
import gym
import gym.wrappers
import gym.wrappers
import gym.wrappers
import gym.wrappers
import numpy as np
import wandb
import torch
import torch.nn as nn
import random
import time
import datetime
from tqdm import tqdm

from rl_models import ActorCritic_Continuous, make_agent

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def strtobool(x):
	if x == 'True':
		return True
	elif x == 'False':
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected')

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
						help='the name of this experiment')
	parser.add_argument('--seed', type=int, default=1,
						help='seed of the experiment')
	# parser.add_argument('--gym-id', type=str, default='HumanoidStandup-v4',
	# 					help='the id of the gym environment')
	parser.add_argument('--gym-id', type=str, default='Ant-v4',
						help='the id of the gym environment')
	parser.add_argument('--total-timesteps', type=int, default=2000000,
						help='total timesteps of the experiments')
	parser.add_argument('--learning-rate', type=float, default=3e-4,
						help='the learning rate of the optimizer')
	parser.add_argument('--num-envs', type=int, default=1,
						help='the number of parallel game environments')
	parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True,
						help='whether to set `torch.backends.cudnn.deterministic=True`')
	parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True,
						help='whether to use cuda devices')
	parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=True,
						help='whether to track and log results on wandb')
	parser.add_argument('--wandb-project-name', type=str, default="ppo",
						help="the wandb's project name")
	parser.add_argument('--wandb-entity', type=str, default=None,
						help="the entity (team) of wandb's project")
	parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=True,
						help='whether to capture videos of the agent performances (check out `videos` folder)')
	parser.add_argument('--num_steps', type=int, default=2048,
						help='number of steps to run the agent in each environment in each rollout')
	parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True,
						help='whether to anneal the learning rate during training')
	parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True,
						help='whether to use Generalized Advantage Estimation')
	parser.add_argument('--gamma', type=float, default=0.99,
						help='the discount factor gamma')
	parser.add_argument('--gae-lambda', type=float, default=0.95,
						help='the lambda for the general advantage estimation')
	parser.add_argument('--num-minibatches', type=int, default=32,
						help='the number of mini-batches')
	parser.add_argument('--update-epochs', type=int, default=10,
						help='the K epochs to update the policy')
	parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True,
						help='whether to normalize the advantage')
	parser.add_argument('--clip-eps', type=float, default=0.2,
						help='the epsilon to use for the surrogate objective')
	parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True,
						help='whether to use a clipped loss for the value function, as per the paper')
	parser.add_argument('--ent-coef', type=float, default=0.0,
						help='coefficient of the entropy')
	parser.add_argument('--vf-coef', type=float, default=0.5,
						help='coefficient of the value function')
	parser.add_argument('--max-grad-norm', type=float, default=0.5,
						help='the maximum norm for the gradient clipping')
	parser.add_argument('--target-kl', type=float, default=None,
						help='the target KL divergence threshold')
	args = parser.parse_args()
	args.batch_size = int(args.num_steps * args.num_envs)
	args.minibatch_size = int(args.batch_size // args.num_minibatches)
	print(args)
	return args

def init_seeds(args):
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.backends.cudnn.deterministic = args.torch_deterministic

def make_continuous_env(gym_id, seed, idx, capture_video, run_name):
	def thunk():
		env = gym.make(gym_id, render_mode='rgb_array')
		env = gym.wrappers.RecordEpisodeStatistics(env)
		if capture_video:
			if idx == 0:
				env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', episode_trigger = lambda x: (x+1) % (200) == 0)
		# env.seed(seed)
		env = gym.wrappers.ClipAction(env)
		env = gym.wrappers.NormalizeObservation(env)
		env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
		env = gym.wrappers.NormalizeReward(env)
		env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
		env.action_space.seed(seed)
		env.observation_space.seed(seed)
		return env
	return thunk


def ppo_continuous_main(args, envs, device):
	agent = make_agent(envs, agent_type='actor-critic-continuous', device=device)
	adam_epsilon = 1e-5
	optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=adam_epsilon)

	obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
	actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
	rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
	dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
	values = torch.zeros((args.num_steps, args.num_envs)).to(device)
	log_probs = torch.zeros((args.num_steps, args.num_envs)).to(device)
	print(f"actions shape: {actions.shape}, rewards shape: {rewards.shape}")

	global_step = 0
	start_time = time.time()
	# print(envs.reset())
	next_obs = torch.Tensor(envs.reset()[0]).to(device)
	next_done = torch.zeros(args.num_envs).to(device)
	num_updates = args.total_timesteps // args.batch_size
	print(num_updates)
	training_bar = tqdm(range(1, num_updates + 1), desc='PPO Training', total=num_updates, unit='updates')

	for update in training_bar:
		if args.anneal_lr:
			frac = 1.0 - (update - 1.0) / num_updates
			optimizer.param_groups[0]["lr"] = frac * args.learning_rate
		
		# Perform steps
		for step in range(0, args.num_steps):
			global_step += 1 * args.num_envs
			obs[step] = next_obs
			dones[step] = next_done
			with torch.no_grad():
				action, log_prob, entropy, value = agent.get_action_and_value(next_obs)
				values[step] = value.flatten()
			log_probs[step] = log_prob
			actions[step] = action

			next_obs, reward, next_done, next_truncated, infos = envs.step(action.cpu().numpy().astype(np.int32))
			rewards[step] = torch.Tensor(reward).to(device).view(-1)
			next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
			# print(infos, rewards.shape, rewards.mean())
			# if infos.get('final_info') is not None:
			# 	for item in infos.get('final_info'):
			# 		if item is not None and "episode" in item.keys():
			# 			if args.track:
			# 				wandb.log({
			# 					"episode/episodic_return": item["episode"]["r"],
			# 					"episode/episodic_length": item["episode"]["l"],
			# 				}, step=global_step)

		# Generalized Advantage Estimataion
		with torch.no_grad():
			next_value = agent.get_value(next_obs).detach().reshape(1, -1)
			if args.gae:
				advantages = torch.zeros_like(rewards).to(device)
				last_gae_lam = 0
				for t in reversed(range(args.num_steps)):
					if t == args.num_steps - 1:
						next_nonterminal = 1.0 - next_done
						next_values = next_value
					else:
						next_nonterminal = 1.0 - dones[t + 1]
						next_value = values[t + 1]
					delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
					advantages[t] = last_gae_lam = delta + args.gae_lambda * args.gamma * next_nonterminal * last_gae_lam
				returns = advantages + values
			else:
				returns = torch.zeros_like(rewards).to(device)
				for t in reversed(range(args.num_steps)):
					if t == args.num_steps - 1:
						next_nonterminal = 1.0 - next_done
						next_return = next_value
					else:
						next_nonterminal = 1.0 - dones[t + 1]
						next_return = returns[t + 1]
					returns[t] = rewards[t] + args.gamma * next_return * next_nonterminal
				advantages = returns - values

		# flatten tensors for batch processing
		batch_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
		batch_log_probs = log_probs.reshape(-1)
		batch_advantages = advantages.reshape(-1)
		batch_returns = returns.reshape(-1)
		batch_actions = actions.reshape((-1, ) + envs.single_action_space.shape)
		batch_values = values.reshape(-1)

		# optimizing policy and value networks
		batch_indexes = np.arange(args.batch_size)
		clip_fracs = []
		for epoch in range(args.update_epochs):
			np.random.shuffle(batch_indexes)
			for start in range(0, args.batch_size, args.minibatch_size):
				end = start + args.minibatch_size
				minibatch_ind = batch_indexes[start:end]
				_, new_logprob, entropy, new_values = agent.get_action_and_value(
					batch_obs[minibatch_ind], batch_actions[minibatch_ind]
				) # action is None
				logratio = new_logprob - batch_log_probs[minibatch_ind]
				ratio = logratio.exp()

				with torch.no_grad():
					# old_approx_kl = (-logratio).mean()
					approx_kl = ((ratio - 1) - logratio).mean()
					clip_fracs += [((ratio - 1.0).abs() > args.clip_eps).float().mean()]

				minibatch_advantages = batch_advantages[minibatch_ind]
				if args.norm_adv:
					minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

				# Policy loss
				pg_loss1 = -ratio * minibatch_advantages
				pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
				pg_loss = torch.max(pg_loss1, pg_loss2).mean()

				# Value loss
				new_values = new_values.view(-1)
				if args.clip_vloss:
					v_loss_unclipped = (new_values - batch_returns[minibatch_ind]) ** 2
					v_clipped = batch_values[minibatch_ind] + torch.clamp(
						new_values - batch_values[minibatch_ind],
						-args.clip_eps,
						args.clip_eps,
					)
					v_loss_clipped = (v_clipped - batch_returns[minibatch_ind]) ** 2
					v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
					v_loss = 0.5 * v_loss_max.mean()
				else:
					v_loss = 0.5 * ((new_values - batch_returns[minibatch_ind]) ** 2).mean()

				# Entropy loss
				entropy_loss = entropy.mean()
				# Minimizing policy and value loss, maximizing entropy loss
				# Maximizing entropy loss encourages exploration
				loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

				# Backpropagation
				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
				optimizer.step()
			if args.target_kl is not None:
				if approx_kl > args.target_kl:
					break

		y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
		var_y = np.var(y_true)
		explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
		# print("actions: ", actions.abs().mean().item())

		if args.track:
			writer_info = {
				"loss/policy_loss": pg_loss.item(),
				"loss/entropy_loss": entropy_loss.item(),
				"loss/value_loss": v_loss.item(),
				"loss/loss": loss.item(),
				"loss/approx_kl": approx_kl.item(),
				"loss/explained_var": explained_var,
				"charts/learning_rate": optimizer.param_groups[0]["lr"],
				"charts/entropy": entropy.mean().item(),
				"loss/clip_frac": np.mean(clip_fracs),
				"rewards/mean_reward": rewards.mean().item(),
				"actions/action_mean_abs": actions.abs().mean().item(),
			}
			wandb.log(writer_info, step=global_step)
	envs.close()
	wandb.finish()
	return agent


if __name__ == '__main__':
	args = parse_args()
	init_seeds(args)
	run_name = f"{args.exp_name}-{args.gym_id}-{current_time}"
	if args.track:
		wandb.init(
			project=args.wandb_project_name,
			entity=None,
			config=vars(args),
			name=run_name,
			monitor_gym=True,
			save_code=True
		)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	envs = gym.vector.SyncVectorEnv([
		make_continuous_env(args.gym_id, args.seed+i, i, args.capture_video, run_name) for i in range(args.num_envs)
	])
	assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

	print("action space single", envs.single_action_space.shape)
	print("observation space shape", envs.single_observation_space.shape)
	observation = envs.reset()
	# for _ in range(200):
	# 	action = envs.action_space.sample()
	# 	step_info = envs.step(action)
	# 	observation, reward, done, truncated, info = step_info
	# 	for item in info:
	# 		if "episode" in item:
	# 			print(item["episode"])
	# print(envs)
	agent = ppo_continuous_main(args, envs, device)
