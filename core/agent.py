import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
import math
import time
import os
import numpy as np

def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size, max_reward, save_render, iter,env_rand):
    if pid > 0:
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
        if hasattr(env, 'np_random'):
            env.np_random.seed(env.np_random.randint(5000) * pid)
        if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
            env.env.np_random.seed(env.env.np_random.randint(5000) * pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = max_reward
    max_reward = -max_reward
    total_c_reward = 0
    min_c_reward = max_reward
    max_c_reward = -max_reward
    num_episodes = 0

    total_e_reward = 0
    min_e_reward = max_reward
    max_e_reward = -max_reward

    if save_render: 
        if not os.path.exists(f'assets/renders/episode_{iter}'):
            os.mkdir(f'assets/renders/episode_{iter}')


    while num_steps < min_batch_size:

        state = env.reset()
        if running_state is not None:
            state = running_state(state)
        reward_episode = 0

        for t in range(10000):
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)

            step_heatmap = False
            if num_episodes == 0:
                step_heatmap = env.heatmap and (render or save_render)
            elif render:
                step_heatmap = ((num_episodes % render == 0) or save_render) and env.heatmap

            next_state, reward, done, _ = env.step(action, step_heatmap=step_heatmap)
            reward_episode += reward

            total_e_reward += reward
            min_e_reward = min(min_e_reward, reward)
            max_e_reward = max(max_e_reward, reward)

            if not save_render:
                if running_state is not None:
                    next_state = running_state(next_state)

                if custom_reward is not None:
                    discrim_reward = custom_reward(state, action)

                    reward = discrim_reward + reward
                    
                    total_c_reward += discrim_reward
                    min_c_reward = min(min_c_reward, reward)
                    max_c_reward = max(max_c_reward, reward)

                mask = 0 if done else 1
        

                memory.push(state, action, mask, next_state, reward)

            if done:
                if save_render:
                    output_file = open(f'assets/renders/episode_{iter}/sample_{num_episodes}.mp4', 'wb')
                    env.render(output_file=output_file)
                if render and num_episodes % render == 0:
                    env.render()
                break

            state = next_state
            if save_render and num_episodes == 5:
                return
            
        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)
    
    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    log['total_e_reward'] = total_e_reward
    log['avg_e_reward'] = total_e_reward / num_steps
    log['max_e_reward'] = max_e_reward
    log['min_e_reward'] = min_e_reward

    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy, device, custom_reward=None, value=None, running_state=None, num_threads=1, max_reward = 1e6, env_rand=2.0):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.running_state = running_state
        self.num_threads = num_threads
        self.max_reward = max_reward
        self.env_rand = env_rand
        self.value = value

    def collect_samples(self, min_batch_size, mean_action=False, render=0, multiprocessing=True, save_render = False, iter=None):
        log = None
        batch = None
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        if multiprocessing:
            thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
            queue = multiprocessing.Queue()
            workers = []

            for i in range(self.num_threads-1):
                worker_args = (i+1, queue, self.env, self.policy, self.custom_reward, mean_action,
                            False, self.running_state, thread_batch_size)
                workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
            for worker in workers:
                worker.start()

            memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, mean_action,
                                        render, self.running_state, thread_batch_size, self.max_reward, save_render, iter, self.env_rand)

            worker_logs = [None] * len(workers)
            worker_memories = [None] * len(workers)
            for _ in workers:
                pid, worker_memory, worker_log = queue.get()
                worker_memories[pid - 1] = worker_memory
                worker_logs[pid - 1] = worker_log
            for worker_memory in worker_memories:
                memory.append(worker_memory)
            batch = memory.sample()
            if self.num_threads > 1:
                log_list = [log] + worker_logs
                log = merge_log(log_list)
            to_device(self.device, self.policy)
            t_end = time.time()
            log['sample_time'] = t_end - t_start
            log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
            log['action_min'] = np.min(np.vstack(batch.action), axis=0)
            log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        else:
            t_start = time.time()

            to_device(torch.device('cpu'), self.policy)

            if not save_render:
                memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, mean_action,
                            render, self.running_state, min_batch_size, self.max_reward, save_render, iter, self.env_rand)
                to_device(self.device, self.policy)
                t_end = time.time()
                batch = memory.sample()

                log['sample_time'] = t_end - t_start
                log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
                log['action_min'] = np.min(np.vstack(batch.action), axis=0)
                log['action_max'] = np.max(np.vstack(batch.action), axis=0)
            else:
                collect_samples(0, None, self.env, self.policy, self.custom_reward, mean_action,
                            render, self.running_state, min_batch_size, self.max_reward, save_render, iter, self.env_rand)

        return batch, log
