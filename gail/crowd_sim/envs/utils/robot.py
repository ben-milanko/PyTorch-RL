from gail.crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
import numpy as np



class Robot(Agent):
    def __init__(self, config, section, relative=False, xy_relative=True):
        super().__init__(config, section)
        self.relative = relative
        self.xy_relative = xy_relative

    def engagement2(self, target):
        r = self.get_full_state()

        pos = np.array([r.px, r.py])
        target = np.array([target.px, target.py])

        dist = np.linalg.norm(target - pos)

        yaw = r.theta % (2 * np.pi)
        R = np.array([[np.cos(yaw), np.sin(yaw)],
                        [-np.sin(yaw), np.cos(yaw)]])

        T_p = target.pos() - self.pos()
        T_p = R.dot(T_p)
        alpha = np.arctan2(T_p[1], T_p[0])
        
        if self.xy_relative:
            return dist*np.cos(alpha), dist*np.sin(alpha)
        else:
            return alpha, dist
        
    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        
        r = self.get_full_state()
        obs = []
        if self.relative:
            for o in ob:
                rel = self.engagement2(o)
                obs.append(ObservableState(rel[0], rel[1], o.vx, o.vy, o.radius))

        state = JointState(r, ob)
        relative=False
        action = self.policy.predict(state)
        return action
