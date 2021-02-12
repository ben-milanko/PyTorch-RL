import logging
import random
import math
import torch
import pickle
from operator import itemgetter

import gym
from gym import error, spaces, utils

from datetime import datetime

import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.policy.policy_factory import policy_factory
from gail.crowd_sim.envs.utils.state import tensor_to_joint_state, JointState
from gail.crowd_sim.envs.utils.action import ActionRot, ActionXY
from gail.crowd_sim.envs.utils.human import Human
from gail.crowd_sim.envs.utils.info import *
from gail.crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, heatmap, device, trajnet):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        # input("hello")
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.nonstop_human = None
        self.centralized_planning = None
        self.centralized_planner = None

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []

        #Two actions for the model
        ACT_DIM = 2
        acthigh = np.array([1]*ACT_DIM)
        self.action_space = spaces.Box(-acthigh,acthigh, dtype=np.float32)

        OBS_DIM = 34
        high = np.array([np.inf]*OBS_DIM)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.phase = None

        #Heatmap
        self.heatmap = heatmap
        self.device = device
        self.values = []

        #Trajnet
        self.frame = 0
        self.global_frame = 0
        self.trajnet = trajnet
        self.trajnet_samples = pickle.load(open(trajnet, 'rb'))

    def configure(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num

        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def generate_human(self, human=None):
        if human is None:
            human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()

        if self.current_scenario == 'circle_crossing':
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * human.v_pref
                py_noise = (np.random.random() - 0.5) * human.v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, -px, -py, 0, 0, 0)

        elif self.current_scenario == 'square_crossing':
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            while True:
                px = np.random.random() * self.square_width * 0.5 * sign
                py = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            while True:
                gx = np.random.random() * self.square_width * 0.5 * - sign
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, gx, gy, 0, 0, 0)

        return human

    def reset(self, phase='test', test_case=None, rand_pos=2.0):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        self.frame = 0

        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                     'val': 0, 'test': self.case_capacity['val']}
        
        rand_start_x = 0
        rand_start_y = 0
        rand_goal_x = 0
        rand_goal_y = 0

        random.seed(datetime.now())
        
        if rand_pos > 0:
            rand_start_x = random.uniform(-rand_pos, rand_pos)
            rand_start_y = random.uniform(-rand_pos, rand_pos)
            rand_goal_x = random.uniform(-rand_pos, rand_pos)
            rand_goal_y = random.uniform(-rand_pos, rand_pos)
        
        # print(rand_start_x, rand_start_y, rand_goal_x, rand_goal_y)

        self.robot.set(rand_start_x, -self.circle_radius+rand_start_y, rand_goal_x, self.circle_radius+rand_goal_y, 0, 0, np.pi / 2)
        if self.case_counter[phase] >= 0:
            np.random.seed(base_seed[phase] + self.case_counter[phase])
            random.seed(base_seed[phase] + self.case_counter[phase])
            if phase == 'test':
                logging.debug('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
            if self.robot.policy is None or not self.robot.policy.multiagent_training and phase in ['train', 'val']:
                # only CADRL trains in circle crossing simulation
                human_num = 1
                self.current_scenario = 'circle_crossing'
            else:
                self.current_scenario = self.test_scenario
                human_num = self.human_num
            self.humans = []
            for _ in range(human_num):
                self.humans.append(self.generate_human())

            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.robot_v = list()
        self.rewards = list()

        self.values = list()

        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        if hasattr(self.robot.policy, 'get_matrix_A'):
            self.As = list()
        if hasattr(self.robot.policy, 'get_feat'):
            self.feats = list()
        if hasattr(self.robot.policy, 'get_X'):
            self.Xs = list()
        if hasattr(self.robot.policy, 'trajs'):
            self.trajs = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = self.compute_observation_for(self.robot)
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError
        # print(ob)

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True, step_heatmap=False):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states)
        else:
            human_actions = []
            for human in self.humans:
                ob = self.compute_observation_for(human)
                human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action[0] #human[0] - action[0]
                vy = human.vy - action[1] #human[1] - action[1]
            else:
                vx = human.vx - action[0] * np.cos(action[1] + self.robot.theta) #action[0] * np.cos(action[1] + self.robot.theta)
                vy = human.vy - action[0] * np.sin(action[1] + self.robot.theta) #action[0] * np.sin(action[1] + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.debug("Collision: distance between robot and p{} is {:.2E} at time {:.2E}".format(human.id, closest_dist, self.global_time))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < (self.robot.radius+1)

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Discomfort(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, 'get_matrix_A'):
                self.As.append(self.robot.policy.get_matrix_A())
            if hasattr(self.robot.policy, 'get_feat'):
                self.feats.append(self.robot.policy.get_feat())
            if hasattr(self.robot.policy, 'get_X'):
                self.Xs.append(self.robot.policy.get_X())
            if hasattr(self.robot.policy, 'traj'):
                self.trajs.append(self.robot.policy.get_traj())

            # update all agents            
            self.robot.step(action)

            # Applying the trajnet movement to humans in the sim
            if self.trajnet:
                sample = self.trajnet_samples[self.global_frame]
                filt = []

                #Sorting the sample to get the 5 closest humans to the robot
                for i in range(len(sample)):
                    dist = np.sqrt(np.square(self.robot.px-float(sample[i][1]))+np.square(self.robot.py-float(sample[i][2])))
                    # if dist < self.robot.sensor_radius:
                    sample[i].append(dist)
                        # sample.append(sample[i])

                sample = sorted(sample, key=itemgetter(3))
                sample = sample[:self.human_num-1]

                
                for i in range(len(self.humans)):
                    # Preserving the position
                    pos_set = False
                    # if self.humans[i].human_id != None:
                    #     for human_sample in sample:
                    #         if human_sample[0] == self.humans[i].human_id:
                    #             self.humans[i].set_position([float(human_sample[1])/2, float(human_sample[2])/2])
                    #             pos_set = True

                    if not pos_set:
                        if len(sample) > i:
                            # Save the trajectory id so the same trajectory can be applied in the next frame
                            self.humans[i].human_id = sample[i][0]
                            new_position = [float(sample[i][1]), float(sample[i][2])]
                            
                            # Calculate & set velocity
                            if sample[i][3] > self.robot.sensor_radius:
                                dx = (new_position[0] - self.robot.px)/sample[i][3] * self.robot.sensor_radius
                                dy = (new_position[1] - self.robot.py)/sample[i][3] * self.robot.sensor_radius
                                new_position = [dx+self.robot.px, dy+self.robot.py]

                            new_velocity = [(new_position[0] - self.humans[i].px)/self.time_step, (new_position[1] - self.humans[i].py)/self.time_step]
                            self.humans[i].set_velocity(new_velocity)

                            self.humans[i].set_position(new_position)
                        else:
                            self.humans[i].set_position([self.robot.sensor_radius+self.robot.px, self.robot.py])
            else:
                for human, action_human in zip(self.humans, human_actions):
                    human.step(action_human)
                    if self.nonstop_human and human.reached_destination():
                        self.generate_human(human)

            self.global_time += self.time_step
            if (self.global_frame >= len(self.trajnet_samples)-1):
                self.frame = 0
                self.global_frame = 0
            else:
                self.frame += 1
                self.global_frame += 1

            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.robot_v.append([self.robot.vx, self.robot.vy])
            self.rewards.append(reward)

            # compute the observation
            ob = self.compute_observation_for(self.robot)
        else:
            ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]

        if self.heatmap and update and step_heatmap:
            fidelity = 5
            actions_1 = range(fidelity)
            actions_2 = range(fidelity)
            states = None
        
            for a1 in actions_1:
                for a2 in actions_2:                            
                    action = None
                    if self.robot.kinematics == 'holonomic':
                        vx = (a1)/(fidelity-1)*2-1
                        vy = (a2)/(fidelity-1)*2-1
                        action = ActionXY(vx, vy)
                    else:
                        speed = (a1)/(fidelity-1)*2-1
                        rotation = (a2)/(fidelity-1)*2-1
                        action = ActionRot(speed, rotation)
                    
                    human_state, reward, done, _ = self.onestep_lookahead(action)
                    r = self.robot.get_next_full_state(action)
                    
                    joint_state = JointState(r, human_state)
                    
                    r_tensor, h_tensor = joint_state.to_tensor()
                    state = torch.hstack([torch.flatten(r_tensor), torch.flatten(h_tensor)])

                    states = state if states == None else torch.vstack((states, state))
            with torch.no_grad():
                self.robot.value.to(torch.device('cpu'))
                vals = self.robot.value(states)
                self.robot.value.to(self.device)
                
                vals = torch.reshape(vals, (fidelity, fidelity))
                self.values.append(vals.numpy())
        
        return ob, reward, done, info

    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            for human in self.humans:
                human_state = human.get_observable_state()

                # Normalise the observations
                human_state.px = human_state.px/self.robot.sensor_radius
                human_state.py = human_state.py/self.robot.sensor_radius
                
                ob.append(human_state)

            ob.append(agent.get_full_state())

        else:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        ####
        # return ob
        return sum([list(o.to_tuple()) for o in ob], [])
        # return sum([list(o) for o in ob])

    def render(self, mode='video', output_file=None, heatmap=False, maximised=True):

        from matplotlib import animation
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.2
        y_offset = 0.4
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'black'
        robot_sensor_color = 'blue'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = False


        if self.heatmap:
            fig, (ax, hm) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
            hm.tick_params(labelsize=12)
            hm.set_xlim(-1, 1)
            hm.set_ylim(-1, 1)
            hm.set_xlabel('x action', fontsize=14)
            hm.set_ylabel('y action', fontsize=14)
            rect = patches.Rectangle((0, 0), 0.3, 0.3, linewidth=1, edgecolor='r', facecolor='none')

        else:
            fig, ax = plt.subplots(figsize=(7, 7))

        ax.tick_params(labelsize=12)
        ax.set_xlim(-21, 21)
        ax.set_ylim(-21, 21)
        ax.set_xlabel('x(m)', fontsize=14)
        ax.set_ylabel('y(m)', fontsize=14)
        show_human_start_goal = False

        # add human start positions and goals
        human_colors = [cmap(i) for i in range(len(self.humans))]
        if show_human_start_goal:
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                            color=human_colors[i],
                                            marker='*', linestyle='None', markersize=8)
                ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i],
                                            marker='o', linestyle='None', markersize=8)
                ax.add_artist(human_start)
        # add robot start position
        robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                    color=robot_color,
                                    marker='o', linestyle='None', markersize=8)
        robot_start_position = [self.robot.get_start_position()[0], self.robot.get_start_position()[1]]
        ax.add_artist(robot_start)
        # add robot and its goal
        robot_positions = [state[0].position for state in self.states]
        goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                color=robot_color, marker='*', linestyle='None',
                                markersize=15, label='Goal')
        robot = plt.Circle(robot_positions[0], self.robot.radius, fill=False, color=robot_color)
        robot_sensor = plt.Circle(robot_positions[0], self.robot.sensor_radius, fill=False, color=robot_sensor_color)


        ax.add_artist(robot)
        ax.add_artist(robot_sensor)
        ax.add_artist(goal)
        ax.legend([robot, goal], ['Robot', 'Goal'], fontsize=10)

        # add humans and their numbers
        human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
        humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=cmap(i))
                    for i in range(len(self.humans))]

        # disable showing human numbers
        if display_numbers:
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i),
                                        color='black') for i in range(len(self.humans))]

        for i, human in enumerate(humans):
            ax.add_artist(human)
            if display_numbers and not self.heatmap:
                ax.add_artist(human_numbers[i])

        # add time annotation
        time = plt.text(0.4, 0.95, 'Time: {}'.format(0), fontsize=10, transform=ax.transAxes)
        reward = plt.text(0.01, 0.95, 'Reward: {}'.format(0), fontsize=10, transform=ax.transAxes)
        reward_sum = plt.text(0.01, 0.9, 'Reward Sum: {}'.format(0), fontsize=10, transform=ax.transAxes)
        action = plt.text(0.01, 0.85, 'Action: [{},{}]'.format(0,0), fontsize=10, transform=ax.transAxes)

        if not self.heatmap:
            ax.add_artist(time)
            ax.add_artist(reward)
            ax.add_artist(reward_sum)
            ax.add_artist(action)

        # compute orientation in each step and use arrow to show the direction
        radius = self.robot.radius
        orientations = []
        for i in range(self.human_num + 1):
            orientation = []
            for state in self.states:
                agent_state = state[0] if i == 0 else state[1][i - 1]
                if self.robot.kinematics == 'unicycle' and i == 0:
                    direction = (
                    (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                        agent_state.py + radius * np.sin(agent_state.theta)))
                else:
                    theta = np.arctan2(agent_state.vy, agent_state.vx)
                    direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                    agent_state.py + radius * np.sin(theta)))
                orientation.append(direction)
            orientations.append(orientation)
            if i == 0:
                arrow_color = 'black'
                arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)]
            else:
                arrows.extend(
                    [patches.FancyArrowPatch(*orientation[0], color=human_colors[i - 1], arrowstyle=arrow_style)])

        for arrow in arrows:
            ax.add_artist(arrow)
        global_step = 0

        if len(self.trajs) != 0:
            human_future_positions = []
            human_future_circles = []
            for traj in self.trajs:
                human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
                                            for step in range(self.robot.policy.planning_depth)]
                                            for i in range(self.human_num)]
                human_future_positions.append(human_future_position)

            for i in range(self.human_num):
                circles = []
                for j in range(self.robot.policy.planning_depth):
                    circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
                    ax.add_artist(circle)
                    circles.append(circle)
                human_future_circles.append(circles)

        def update(frame_num):
            nonlocal global_step
            nonlocal arrows
            # nonlocal rect
            
            global_step = frame_num
            robot.center = robot_positions[frame_num]
            robot_sensor.center = robot_positions[frame_num]

            for i, human in enumerate(humans):
                human.center = human_positions[frame_num][i]
                if display_numbers:
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] + y_offset))
            for arrow in arrows:
                arrow.remove()

            for i in range(self.human_num + 1):
                orientation = orientations[i]
                if i == 0:
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color='black',
                                                        arrowstyle=arrow_style)]
                else:
                    arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=cmap(i - 1),
                                                            arrowstyle=arrow_style)])

            for arrow in arrows:
                ax.add_artist(arrow)
                
            time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
            reward.set_text('Reward: {:.2f}'.format(self.rewards[frame_num]))
            reward_sum.set_text('Reward Sum: {:.2f}'.format(sum(self.rewards[0:frame_num])))
            action.set_text('Action: [{:.2f},{:.2f}]'.format(self.robot_v[frame_num][0],self.robot_v[frame_num][1]))

            if self.heatmap:
                rect.remove()
                rect = patches.Rectangle((self.robot_v[frame_num][0], self.robot_v[frame_num][1]), 0.3, 0.3, linewidth=1, edgecolor='r', facecolor='none')
                hm.add_patch(rect)

                hm.imshow(self.values[frame_num], cmap='hot', interpolation='nearest')

            if len(self.trajs) != 0:
                for i, circles in enumerate(human_future_circles):
                    for j, circle in enumerate(circles):
                        circle.center = human_future_positions[global_step][i][j]

        anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step*500, blit=False, repeat=True)
        anim.running = True

        #output_file = None #open('assets/test.gif', 'wb')

        if output_file is not None:
            # save as video
            if heatmap:
                writer = animation.FFMpegWriter(fps=12, metadata=dict(artist='Me'), bitrate=1800)
            else:
                writer = animation.ImageMagickWriter(fps=12,metadata=dict(artist='Me'),bitrate=1800, codec='libx264')
            anim.save(output_file.name, writer=writer)

            output_file.close()
        else:
            if heatmap:
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
            plt.show()
