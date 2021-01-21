import argparse
import logging
import gym
import os
import sys
import pickle
import time
import importlib
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

from crowd_sim.envs.utils.agent import BasicRobot
# from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.configs.icra_benchmark import gail

from crowd_sim.envs.utils.robot import Robot

from crowd_sim.envs.crowd_sim import CrowdSim


parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="CrowdSim-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--save-render', action='store_true', default=False,
                    help='Save and log the runs as gifs')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-reward', type=float, default=1e6, metavar='G',
                    help='Limits reward per step and cumulative reward (default: 1e6)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--multiprocessing', action='store_true', default=False,
                    help="Sets multiprocessing to true")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--no-wandb', action='store_true', default=False,
                    help='log run on weights & biases')
# parser.add_argument('--pre-train', action='store_true', default=False,
#                     help='pretrains the policy from imitation learning on orca')
parser.add_argument('--wandb-description', default='', metavar='G',
                    help='description to append to wandb run title')
args = parser.parse_args()

expert_name = args.expert_traj_path.split('/')[-1].split('.')[0]

if not args.no_wandb: wandb.init(project='crowd_rl', name=f'gail_steps_{args.max_iter_num}_{expert_name}_{args.wandb_description}',)


dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = None
robot = None
if args.env_name == 'CrowdSim-v0':
    env = CrowdSim()
    env_config = gail.EnvConfig(True)
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.time_step = env.time_step
    #robot.set_policy(policy)
    robot = BasicRobot()

    env.set_robot(robot)
else:
    gym.make(args.env_name)

state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]
running_state = None #ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    value_net = Value(state_dim)
    discrim_net = Discriminator(state_dim + action_dim)
else:
    policy_net, value_net, discrim_net = pickle.load(open(args.model_path, "rb"))

robot.set_act(lambda x : policy_net(tensor(x))[0][0].numpy())

discrim_criterion = nn.BCELoss()

if not args.no_wandb: 
    wandb.watch(policy_net)
    wandb.watch(value_net)
    wandb.watch(discrim_net)

to_device(device, policy_net, value_net, discrim_net, discrim_criterion)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 10
optim_batch_size = 64

# load trajectory
# expert_traj, running_state = pickle.load(open("/home/rhys/CrowdNavigation/PyTorch-RL/expert_traj (1).p", "rb"))
# running_state.fix = True
# expert_traj, running_state = pickle.load(open("/home/rhys/CrowdNavigation/PyTorch-RL/assets/expert_traj/Hopper-v2_expert_traj.p", "rb"))
expert_traj = pickle.load(open("expert_traj.p", "rb"))


def expert_reward(state, action):
    state_action = tensor(np.hstack([state, action]), dtype=dtype)
    with torch.no_grad():
        return -math.log(discrim_net(state_action)[0].item())


"""create agent"""
agent = Agent(env, policy_net, device, custom_reward=expert_reward,
              running_state=running_state, num_threads=args.num_threads, max_reward=args.max_reward)


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """update discriminator"""
    for _ in range(1):
        expert_state_actions = torch.from_numpy(expert_traj).to(dtype).to(device)
        g_o = discrim_net(torch.cat([states, actions], 1))
        e_o = discrim_net(expert_state_actions)
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
            discrim_criterion(e_o, zeros((expert_state_actions.shape[0], 1), device=device))
        discrim_loss.backward()
        optimizer_discrim.step()

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render, multiprocessing=args.multiprocessing)
        discrim_net.to(device)
        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()
        """evaluate with determinstic action (remove noise for exploration)"""
        discrim_net.to(torch.device('cpu'))
        _, log_eval = agent.collect_samples(args.eval_batch_size, multiprocessing=args.multiprocessing, mean_action=True)
        discrim_net.to(device)
        t2 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\ttrain_discrim_R_avg {:.2f}\ttrain_R_avg {:.2f}\teval_discrim_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['avg_c_reward'], log['avg_reward'], log_eval['avg_c_reward'], log_eval['avg_reward']))
        
        log_combine = {**log, **log_eval}
        if not args.no_wandb: wandb.log(log_combine)        

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if args.save_render:
                print(f'Saving renders to: assets/renders/episode_{i_iter+1}')
                agent.collect_samples(args.min_batch_size, render=args.render, multiprocessing=args.multiprocessing, save_render=True, iter=i_iter+1)
                if not args.no_wandb: wandb.log({f'episode_{i_iter+1}': [wandb.Video(f'assets/renders/episode_{i_iter+1}/sample_{i}.gif', fps=12, format="gif") for i in range(5)]})
            print('Saving model to: ' + os.path.join(f'assets/learned_models/{args.env_name}_gail{i_iter+1}.p'))
            to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
            pickle.dump((policy_net, value_net, discrim_net), open(f'assets/learned_models/{args.env_name}_gail{i_iter+1}.p', 'wb'))
            to_device(device, policy_net, value_net, discrim_net)

            if not args.no_wandb: wandb.save(f'assets/learned_models/{args.env_name}_gail{i_iter+1}.p')


        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
