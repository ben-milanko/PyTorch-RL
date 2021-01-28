import abc
import logging
import numpy as np
from numpy.linalg import norm
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState


class Agent(object):
    def __init__(self, config, section, max_rot=1):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = getattr(config, section).visible
        self.v_pref = getattr(config, section).v_pref
        self.radius = getattr(config, section).radius
        self.policy = policy_factory[getattr(config, section).policy]()
        self.sensor = getattr(config, section).sensor
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None
        self.max_rot = max_rot

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        if self.time_step is None:
            raise ValueError('Time step is None')
        policy.set_time_step(self.time_step)
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_vx = action.v * np.cos(self.theta)
            next_vy = action.v * np.sin(self.theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_start_position(self):
        return self.sx, self.sy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if type(action) is np.ndarray:
            assert action.shape[0] == 2

        else:
            if self.kinematics == 'holonomic':
                assert isinstance(action, ActionXY)
            else:
                assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action[0] * delta_t
            py = self.py + action[1] * delta_t
        else:
            theta = self.theta + action[1]*self.max_rot
            px = self.px + np.cos(theta) * action[0] * delta_t
            py = self.py + np.sin(theta) * action[0] * delta_t

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if type(action) is np.ndarray:
            if self.kinematics == 'holonomic':
                self.vx = action[0]
                self.vy = action[1]
            else:
                self.theta = (self.theta + action[1]*self.max_rot) % (2 * np.pi)
                self.vx = action[0] * np.cos(self.theta)
                self.vy = action[0] * np.sin(self.theta)
        else:
            if self.kinematics == 'holonomic':
                self.vx = action.vx
                self.vy = action.vy
            else:
                self.theta = (self.theta + action.r*self.max_rot) % (2 * np.pi)
                self.vx = action.v * np.cos(self.theta)
                self.vy = action.v * np.sin(self.theta)

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

class BasicPolicy():
    def __init__(self):
        self.multiagent_training = False

class BasicRobot(Agent):
    def __init__(self):
        
        self.visible = True #getattr(config, section).visible
        self.v_pref = 1 #getattr(config, section).v_pref
        self.radius = 0.3 #getattr(config, section).radius
        self.policy = BasicPolicy() #policy_factory[getattr(config, section).policy]()
        self.sensor = "coordinates" #getattr(config, section).sensor
        self.kinematics = "holonomic" #self.policy.kinematics if self.policy is not None else None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

    def set_act(self, action_function):
        self.action_function = action_function
        
    def act(self, ob):
        return self.action_function(ob)



    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        # print("action", action)
        pos = self.compute_position(action, self.time_step)
        # print("pos", pos)
        self.px, self.py = pos
        if type(action) is np.ndarray:
            if self.kinematics == 'holonomic':
                self.vx = action[0]
                self.vy = action[1]
            else:
                self.theta = (self.theta + action[1]) % (2 * np.pi)
                self.vx = action[0] * np.cos(self.theta)
                self.vy = action[0] * np.sin(self.theta)
        else:
            if self.kinematics == 'holonomic':
                self.vx = action.vx
                self.vy = action.vy
            else:
                self.theta = (self.theta + action.r) % (2 * np.pi)
                self.vx = action.v * np.cos(self.theta)
                self.vy = action.v * np.sin(self.theta)
