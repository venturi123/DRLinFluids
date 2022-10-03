import gym
import os
import re
import shutil
from abc import abstractmethod
from time import time
from scipy import signal
import numpy as np
import pandas as pd
from drlinfluids import runner, utils
from gym import spaces


class OpenFoam(gym.Env):
	"""The main DRLinFluids Gym class.
    It encapsulates an environment with arbitrary behind-the-scenes dynamics.
    An environment can be partially or fully observed.

    Parameters
    ----------
    foam_root_path : str
        Path to simulation file.
    foam_params : list
        simulation parameters.
    agent_params : list
        DRL parameters.
    state_params : list
        Running dimension.

    Examples
    --------
    from DRLinFluids.environments_tianshou import OpenFoam_tianshou

    note
    --------
    The main API methods that users of this class need to know are:
    - :meth:`step` - Takes a step in the environment using an action returning the next observation, reward,
    if the environment terminated and more information.
    - :meth:`reward_function` - Define reward_funtion and calculate.
    - :meth:`reset` - Resets the environment to an initial state, returning the initial observation.
    """
	def __init__(self, foam_root_path, foam_params, agent_params, state_params,  logger_params=None):
		super().__init__()
		self.foam_params = foam_params
		self.agent_params = agent_params
		self.state_params = state_params
		self.foam_root_path = foam_root_path
		self.state_params['probe_info'] = utils.read_foam_file(
			'/'.join([foam_root_path, 'system', 'probes'])
		)
		self.logger_params = logger_params
		self.dashboard_data = {}
		self.trajectory_start_time = 0
		self.trajectory_end_time = 0
		self.num_episode = 0
		self.info_list = []
		self.episode_reward_sequence = []
		self.exec_info = {}
		self.num_trajectory = 0
		self.trajectory_reward = np.array([])
		self.all_episode_trajectory_reward = pd.DataFrame()
		self.state_data = np.array([])
		self.episode_reward = 0
		self.actions_sequence = np.array([])
		self.start_actions = 0
		self.end_actions = 0
		self.single_step_actions = np.array([])
		self.all_episode_actions = pd.DataFrame()
		self.all_episode_decorated_actions = pd.DataFrame()
		self.all_episode_single_step_actions = pd.DataFrame()
		self.probe_velocity_df = pd.DataFrame()
		self.probe_pressure_df = pd.DataFrame()
		self.force_df = pd.DataFrame()
		self.force_Coeffs_df = pd.DataFrame()
		self.history_force_df = pd.DataFrame()
		self.history_force_Coeffs_df = pd.DataFrame()
		self.cfd_init_time_str = str(float(foam_params['cfd_init_time'])).rstrip('0').rstrip('.')
		self.decimal = int(np.max([
			len(str(agent_params['interaction_period']).split('.')[-1]),
			len(str(foam_params['cfd_init_time']).split('.')[-1])
		]))

		if server:
			runner.run_init(foam_root_path, foam_params)
			self.velocity_table_init = utils.read_foam_file(
				foam_root_path + f'/postProcessing/probes/0/U',
				dimension=self.foam_params['num_dimension']
			)
			self.pressure_table_init = utils.read_foam_file(
				foam_root_path + f'/postProcessing/probes/0/p',
				dimension=self.foam_params['num_dimension']
			)

		if self.state_params['type'] == 'pressure':
			self.state_space = spaces.Box(low=-np.Inf, high=np.Inf,
			                              shape=(int(self.state_params['probe_info'].shape[0]),), dtype=np.float32)
		elif self.state_params['type'] == 'velocity':
			if self.foam_params['num_dimension'] == 2:
				self.state_space = spaces.Box(low=-np.Inf, high=np.Inf,
				                              shape=(2 * int(self.state_params['probe_info'].shape[0]),),
				                              dtype=np.float32)
			elif self.foam_params['num_dimension'] == 3:
				self.state_space = spaces.Box(low=-np.Inf, high=np.Inf,
				                              shape=(3 * int(self.state_params['probe_info'].shape[0]),),
				                              dtype=np.float32)
			else:
				assert 0, 'Simulation type error'
		else:
			assert 0, 'No define state type error'

		self.action_space = spaces.Box(self.agent_params['minmax_value'][0], self.agent_params['minmax_value'][1],
		                               shape=(len(self.agent_params['variables_q0']),), dtype=np.float32)
		self.seed()

	def step(self,actions: np.ndarray):
		"""Run one timestep of the environment's dynamics."""
		self.trajectory_start_time = time()
		self.num_trajectory += 1
		if actions is None:
			print("carefully, no action given; by default, no jet!")

		self.actions_sequence = np.append(self.actions_sequence, actions)

		if self.num_trajectory < 1.5:
			self.start_actions = [0]
			self.end_actions = [self.actions_sequence[0]]
		else:
			self.start_actions = [self.actions_sequence[-2]]
			self.end_actions = [self.actions_sequence[-1]]

		start_time_float = np.around(
			float(self.cfd_init_time_str) + (self.num_trajectory - 1) * self.agent_params['interaction_period'],
			decimals=self.decimal
		)
		end_time_float = np.around(start_time_float + self.agent_params['interaction_period'], decimals=self.decimal)

		start_time_filename, start_time_path = utils.get_current_time_path(self.foam_root_path)

		utils.dict2foam(
			start_time_path,
			utils.actions2dict(self.agent_params['entry_dict_q0'], self.agent_params['variables_q0'],
			                   self.start_actions)
		)

		utils.dict2foam(
			start_time_path,
			utils.actions2dict(self.agent_params['entry_dict_q1'], self.agent_params['variables_q1'], self.end_actions)
		)

		start_time = [start_time_float]
		utils.dict2foam(
			start_time_path,
			utils.actions2dict(self.agent_params['entry_dict_t0'], self.agent_params['variables_t0'], start_time)
		)

		simulation_start_time = time()
		cfd.run(
			self.foam_root_path,
			self.foam_params,
			self.agent_params['writeInterval'],
			self.agent_params['deltaT'],
			start_time_float, end_time_float
		)
		simulation_end_time = time()

		self.probe_velocity_df = utils.read_foam_file(
			self.foam_root_path + f'/postProcessing/probes/{start_time_filename}/U',
			dimension=self.foam_params['num_dimension']
		)

		self.probe_pressure_df = utils.read_foam_file(
			self.foam_root_path + f'/postProcessing/probes/{start_time_filename}/p',
			dimension=self.foam_params['num_dimension']
		)

		self.force_df = utils.resultant_force(
			utils.read_foam_file(
				self.foam_root_path + f'/postProcessing/forcesIncompressible/{start_time_filename}/forces.dat'
			)
		)

		self.force_Coeffs_df = utils.read_foam_file(
			self.foam_root_path + f'/postProcessing/forceCoeffsIncompressible/{start_time_filename}/forceCoeffs.dat'
		)

		if self.num_trajectory < 1.5:
			self.history_force_df = self.force_df
			self.history_force_Coeffs_df = self.force_Coeffs_df
		else:
			self.history_force_df = pd.concat([self.history_force_df, self.force_df[1:]]).reset_index(drop=True)
			self.history_force_Coeffs_df = pd.concat(
				[self.history_force_Coeffs_df, self.force_Coeffs_df[1:]]
			).reset_index(drop=True)

		if self.state_params['type'] == 'pressure':
			next_state = self.probe_pressure_df.iloc[-1, 1:].to_numpy()
		elif self.state_params['type'] == 'velocity':
			next_state = self.probe_velocity_df.iloc[-1, 1:].to_numpy()
		else:
			next_state = False
			assert next_state, 'No define state type'
		self.state_data = np.append(self.state_data, next_state)

		reward = self.reward_function()
		# print(self.num_trajectory, self.start_actions, self.end_actions,reward)
		self.trajectory_reward = np.append(self.trajectory_reward, reward)
		self.episode_reward += reward

		terminal = False

		self.trajectory_end_time = time()

		self.exec_info = {
			'episode': self.num_episode,
			'trajectory': self.num_trajectory,
			'start_time_float': start_time_float,
			'end_time_float': end_time_float,
			'timestampStart': self.trajectory_start_time,
			'timestampEnd': self.trajectory_end_time,
			'current_trajectory_reward': reward,
			'episode_reward': self.episode_reward,
			'actions': actions,
			'cfd_running_time': simulation_end_time - simulation_start_time,
			'number_cfd_timestep': int(np.around((end_time_float - start_time_float) / self.foam_params['delta_t'])),
			'envName': self.foam_root_path.split('/')[-1],
			'current_state': self.state_data[-2],
			'next_state': next_state,
		}
		self.info_list.append(self.exec_info)

		return next_state, reward, terminal, {}


	@abstractmethod
	def reward_function(self):
		pass

	def reset(self):
		"""Resets the environment to an initial state and returns the initial observation."""
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
			self.info_list = pd.DataFrame(self.info_list)
			self.info_list.to_csv(
				log_dir + f'/info_list_{self.num_episode}.csv'
			)
		self.num_episode += 1
		self.num_trajectory = 0
		self.episode_reward = 0
		self.actions_sequence = []
		self.trajectory_reward = []
		self.decorated_actions_sequence = []
		self.history_force_df = pd.DataFrame()
		self.history_force_Coeffs_df = pd.DataFrame()
		self.info_list = []
		for f_name in os.listdir(self.foam_root_path):
			if re.search(r'^\d+\.?\d*', f_name):
				if (f_name != '0') and (f_name != self.cfd_init_time_str):
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
