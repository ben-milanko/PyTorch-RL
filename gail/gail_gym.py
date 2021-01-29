import argparse
import logging
import gym
import os
import sys
import pickle
import time
import importlib
import wandb
import socket

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

from gail.crowd_sim.envs.utils.agent import BasicRobot
from gail.crowd_sim.configs.icra_benchmark import gail
from gail.crowd_sim.envs.crowd_sim import CrowdSim


parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="CrowdSim-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', default="starting_assets/rgl_expert_traj.p", metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--save-render', action='store_false', default=True,
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
parser.add_argument('--max-iter-num', type=int, default=20000, metavar='N',
                    help='maximal number of main iterations (default: 20000)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=100, metavar='N',
                    help="interval between saving model (default: 100, 0 means don't save)")
parser.add_argument('--multiprocessing', action='store_true', default=False,
                    help="Sets multiprocessing to true")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--model-path', default="starting_assets/gail_model.p", metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--no-wandb', action='store_true', default=False,
                    help='log run on weights & biases')
parser.add_argument('--holonomic', action='store_true', default=False,
                    help='Run holonomic movement (default: unicyle)')
parser.add_argument('--wandb-description', default='', metavar='G',
                    help='description to append to wandb run title')
parser.add_argument('--env-rand', type=float, default=2.0, metavar='N',
                    help='additional environmental randomness to start and end positions')
parser.add_argument('--robot-rot', type=float, default=np.pi/10, metavar='N',
                    help='robot rotation speed factor (default: np.pi/10)')
parser.add_argument('--relative', default='xy',
                    help='Train agent on relative position of agents, options are [xy] and [polar], anything else will be none')
parser.add_argument('--reverse', action='store_true', default=False,
                    help='Allow robot to reverse in unicycle mode (default: False)')
args = parser.parse_args()

expert_name = args.expert_traj_path.split('/')[-1].split('.')[0]
starting_model = args.model_path.split('/')[-1].split('.')[0]
kinematics = 'holonomic' if args.holonomic else 'unicycle'
tags = [
        f'steps:{args.max_iter_num}',
        f'expert:{expert_name}',
        f'kinematics:{kinematics}',
        f'reward:mixed',
        f'relative:{args.relative}',
        f'rotation_clamp:{args.robot_rot:0.2f}',
        f'reverse:{args.reverse}',
        f'action_clamp:tanh',
        f'goal_randomisation:{args.env_rand}',
        f'starting_model:{starting_model}',
        f'system:{socket.gethostname()}'
    ]
print(f'Tags: {tags}')
if not args.no_wandb: wandb.init(project='crowd_rl', name=f'GAIL {args.wandb_description}',tags=tags)


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
    
    relative_xy = False
    relative = False

    if args.relative == 'xy':
        relative = True
        relative_xy = True
    elif args.relative == 'polar':
        relative = True
        relative_xy = False

    robot = BasicRobot(relative=relative, relative_xy=relative_xy, max_rot=args.robot_rot, kinematics=kinematics, reverse=args.reverse)
    env.set_robot(robot)

else:
    gym.make(args.env_name)

state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]
running_state = None #ZFilter((state_dim,), clip=5)

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

# if not args.no_wandb: 
#     wandb.watch(policy_net)
#     wandb.watch(value_net)
#     wandb.watch(discrim_net)

to_device(device, policy_net, value_net, discrim_net, discrim_criterion)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 10
optim_batch_size = 64

# load trajectory
expert_traj = pickle.load(open("expert_traj.p", "rb"))

def expert_reward(state, action):
    state_action = tensor(np.hstack([state, action]), dtype=dtype)
    with torch.no_grad():
        return -math.log(discrim_net(state_action)[0].item())


"""create agent"""
agent = Agent(env, policy_net, device, custom_reward=expert_reward,
              running_state=running_state, num_threads=args.num_threads, max_reward=args.max_reward, env_rand=args.env_rand)


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
                if not args.no_wandb: wandb.log({f'episode_{(i_iter+1):0{len(str(args.max_iter_num))}}': [wandb.Video(f'assets/renders/episode_{i_iter+1}/sample_{i}.gif', fps=12, format="gif") for i in range(5)]})
            
            print('Saving model to: ' + os.path.join(f'assets/learned_models/{args.env_name}_gail{i_iter+1}.p'))
            to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
            pickle.dump((policy_net, value_net, discrim_net), open(f'assets/learned_models/{args.env_name}_gail{i_iter+1}.p', 'wb'))
            to_device(device, policy_net, value_net, discrim_net)

            if not args.no_wandb: wandb.save(f'assets/learned_models/{args.env_name}_gail{i_iter+1}.p')

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
