import os
import re
import shutil
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from drlinfluids import runner, utils

from tensorforce.environments import Environment


class OpenFoam(Environment, metaclass=ABCMeta):
    def __init__(self, foam_root_path, foam_params, agent_params, state_params, logger_params=None):
        super().__init__()
        self.foam_root_path = foam_root_path
        self.foam_params = foam_params
        self.agent_params = agent_params
        self.state_params = state_params
        self.state_params['probe_info'] = utils.read_foam_file(
            '/'.join([foam_root_path, 'system', 'probes'])
        )
        self.logger_params = logger_params
        self.dashboard_data = {}
        self.start_time_filename = ''
        self.start_time_path = ''
        self.trajectory_start_time = 0
        self.trajectory_end_time = 0
        self.num_episode = 0
        self.info_list=[]
        self.episode_reward_sequence = []
        self.exec_info = {}
        self.num_trajectory = 0
        self.trajectory_reward = np.array([])
        self.all_episode_trajectory_reward = pd.DataFrame()
        self.state_data = np.array([])
        self.episode_reward = 0
        self.decorated_actions = np.array([])
        self.actions_sequence = np.array([])
        self.decorated_actions_sequence = np.array([])
        self.start_actions = 0
        self.end_actions = 0
        self.single_step_actions = np.array([])
        self.all_episode_actions = pd.DataFrame()
        self.all_episode_decorated_actions = pd.DataFrame()
        self.all_episode_single_step_actions = pd.DataFrame()
        self.probe_velocity = pd.DataFrame()
        self.probe_pressure = pd.DataFrame()
        self.force = pd.DataFrame()
        self.force_Coeffs = pd.DataFrame()
        self.history_force = pd.DataFrame()
        self.history_force_Coeffs = pd.DataFrame()
        self.cfd_init_time = str(float(foam_params['cfd_init_time'])).rstrip('0').rstrip('.')
        self.decimal = int(np.max([
                    len(str(agent_params['interaction_period']).split('.')[-1]),
                    len(str(foam_params['cfd_init_time']).split('.')[-1])
                ]))


        runner.run_init(foam_root_path, foam_params)
        self.velocity_table_init = utils.read_foam_file(
            foam_root_path + f'/postProcessing/probes/0/U',
            dimension=self.foam_params['num_dimension']
        )
        self.pressure_table_init = utils.read_foam_file(
            foam_root_path + f'/postProcessing/probes/0/p',
            dimension=self.foam_params['num_dimension']
        )


    @abstractmethod
    def states(self):
        pass


    @abstractmethod
    def actions(self):
        pass


    @abstractmethod
    def execute(self, actions=None):
        pass


    @abstractmethod
    def reward_function(self):
        pass


    def reset(self):
        if self.logger_params:
            if 'log_dir' in self.logger_params:
                log_dir = self.logger_params['log_dir']
            else:
                log_dir = self.foam_root_path + '/logs'

            if self.num_episode < 0.5:
                os.makedirs(log_dir)
            else:
                self.episode_reward_sequence.append(self.episode_reward)
                pd.DataFrame(
                    self.episode_reward_sequence
                ).to_csv(log_dir + '/total_reward.csv', index=False, header=False)
                if self.episode_reward_sequence[-1] == np.max(self.episode_reward_sequence):
                    pd.DataFrame(
                        self.actions_sequence
                    ).to_csv(log_dir + '/best_actions.csv', index=False, header=False)
                    # self.history_force_Coeffs_df[:, 3]=signal.savgol_filter(self.force_Coeffs_df.iloc[:, 3], 11, 3)
                    pd.DataFrame(
                        self.history_force_Coeffs
                    ).to_csv(log_dir + '/best_history_force_Coeffs_df.csv', index=False, header=False)
                    with open(log_dir + '/info.txt', 'w') as f:
                        f.write(f'Current number of best reward episode is {self.num_episode}')

            if self.num_episode == 1:
                self.all_episode_actions = pd.DataFrame(self.actions_sequence)
                self.all_episode_decorated_actions = pd.DataFrame(self.decorated_actions_sequence)
                self.all_episode_trajectory_reward = pd.DataFrame(self.trajectory_reward)
            else:
                self.all_episode_actions[self.num_episode - 1] = pd.DataFrame(self.actions_sequence)
                self.all_episode_decorated_actions[self.num_episode - 1] = pd.DataFrame(self.decorated_actions_sequence)
                self.all_episode_trajectory_reward[self.num_episode - 1] = pd.DataFrame(self.trajectory_reward)

            self.all_episode_actions.to_csv(
                log_dir + '/all_episode_actions.csv', index=False, header=False
            )
            self.all_episode_trajectory_reward.to_csv(
                log_dir + '/all_episode_trajectory_reward.csv', index=False, header=False
            )
            self.history_force_Coeffs.to_csv(
                log_dir + f'/history_force_Coeffs_df_{self.num_episode}.csv', index=False, header=False
            )
            self.info_list=pd.DataFrame(self.info_list)
            self.info_list.to_csv(
                log_dir + f'/info_list_{self.num_episode}.csv'
            )

        self.num_episode += 1
        self.num_trajectory = 0
        self.episode_reward = 0
        self.actions_sequence = []
        self.trajectory_reward = []
        self.decorated_actions_sequence = []
        self.history_force = pd.DataFrame()
        self.history_force_Coeffs = pd.DataFrame()
        self.info_list=[]
        for f_name in os.listdir(self.foam_root_path):
            if re.search(r'^\d+\.?\d*', f_name):
                if (f_name != '0') and (f_name != self.cfd_init_time):
                    shutil.rmtree('/'.join([self.foam_root_path, f_name]))
            elif f_name == 'postProcessing':
                shutil.rmtree('/'.join([self.foam_root_path, f_name]))
            else:
                pass

        if self.state_params['type'] == 'pressure':
            init_state = self.pressure_table_init.iloc[-1, 1:].to_numpy()
        elif self.state_params['type'] == 'velocity':
            init_state = self.velocity_table_init.iloc[-1, 1:].to_numpy()
        else:
            init_state = False
            assert init_state, 'No define state type'

        self.state_data = np.append(self.state_data, init_state)

        return init_state


    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


    def close(self):
        super().close()

