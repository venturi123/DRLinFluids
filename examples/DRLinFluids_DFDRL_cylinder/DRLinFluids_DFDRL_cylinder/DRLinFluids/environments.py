from typing import Any, Callable, List, Optional, Tuple, Union

import gym
import numpy as np
import os
import re
import shutil
from abc import ABCMeta, abstractmethod
from time import time
from scipy import signal
import numpy as np
import pandas as pd
from DRLinFluids import cfd, utils
from gym import spaces, logger
from gym.utils import seeding
from tianshou.utils import RunningMeanStd
from sklearn.preprocessing import StandardScaler


class OpenFoam(gym.Env):
	def __init__(self,
	             foam_root_path:Optional[str]= None,
	              foam_params: Optional[dict]= None,
	             agent_params: Optional[dict]= None,
	             state_params: Optional[dict]= None,
	             server=True,**kwargs):
		self.foam_params = foam_params
		self.agent_params = agent_params
		self.state_params = state_params
		self.foam_root_path = foam_root_path
		self.task = 'OpenFoam-v0'
		self.state_params['probe_info'] = utils.read_foam_file(
			'/'.join([foam_root_path, 'system', 'probes'])
		)
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
		self.decorated_actions = np.array([])
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
		self.initial_force_Coeffs_df = pd.DataFrame()
		self.history_force_Coeffs_df = pd.DataFrame()
		self.history_force_Coeffs_df_alltime = pd.DataFrame()
		self.history_force_Coeffs_df_stepnumber=0
		self.start_time_float=0
		self.end_time_float=0
		self.action_time = 0
		self.vortex_shedding = 0
		self.svd_rank_df=10

		self.cfd_init_time_str = str(float(foam_params['cfd_init_time'])).rstrip('0').rstrip('.')
		self.decimal = int(np.max([
			len(str(agent_params['interaction_period']).split('.')[-1]),
			len(str(foam_params['cfd_init_time']).split('.')[-1])
		]))
		self.pressure_DMD_initial_snapshot=np.array([])
		self.control_matrix_gammaDMDc=np.array([])

		if server:
			action_tocsv_list = [[0, 0, 0, 0],
				[self.foam_params['cfd_init_time'], 0, 0, 0]]
			pd.DataFrame(
					action_tocsv_list
				).to_csv(self.foam_root_path + '/system/jet.csv', index=False, header=False)
			for f_name in os.listdir(self.foam_root_path):
				if f_name == 'prosessor0':
					shutil.rmtree('/'.join([self.foam_root_path, f_name]))
				elif f_name == 'prosessor1':
					shutil.rmtree('/'.join([self.foam_root_path, f_name]))
				elif f_name == 'prosessor2':
					shutil.rmtree('/'.join([self.foam_root_path, f_name]))
				elif f_name == 'prosessor3':
					shutil.rmtree('/'.join([self.foam_root_path, f_name]))
				else:
					pass
			cfd.run_init(foam_root_path, foam_params)

			self.velocity_table_init = utils.read_foam_file(
				foam_root_path + f'/postProcessing/probes/0.000/U',
				dimension=self.foam_params['num_dimension']
			)
			cfd_init_time = int(self.foam_params['cfd_init_time'] - self.agent_params['interaction_period'])
			self.pressure_table_init = utils.read_foam_file(
				foam_root_path + f'/postProcessing/probes/0.000/p',
				dimension=self.foam_params['num_dimension']
			)
			self.initial_force_Coeffs_df= utils.read_foam_file(
				foam_root_path + f'/forceCoeffsIncompressible/0.000/forceCoeffs.dat'
		    )
			self.pressure_DMD_initial = utils.read_foam_file(
				foam_root_path + f'/probes/0.000/p',
				dimension=self.foam_params['num_dimension']
			)
			self.pressure_DMD_initial = self.pressure_DMD_initial.iloc[1:, 1:].to_numpy().T
			self.pressure_DMD_initial_snapshot=self.pressure_DMD_initial[:,(int(self.agent_params['interaction_period'] / self.foam_params['delta_t'])-1)
																		   :(int(self.agent_params['interaction_period'] / self.foam_params['delta_t'])*30):
																		   (int(self.agent_params['interaction_period'] / self.foam_params['delta_t']))]

			self.pressure_DMD_initial_snapshot_init=self.pressure_DMD_initial_snapshot

		self.dmd_state=np.zeros((int(self.state_params['probe_info'].shape[0]+1),
								 30), dtype=int)
		if self.state_params['type'] == 'pressure':
			self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
			                              shape=(self.dmd_state.shape), dtype=np.float32)
		elif self.state_params['type'] == 'velocity':
			if self.foam_params['num_dimension'] == 2:
				self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
				                              shape=(2 * int(self.state_params['probe_info'].shape[0]),),
				                              dtype=np.float32)
			elif self.foam_params['num_dimension'] == 3:
				self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
				                              shape=(3 * int(self.state_params['probe_info'].shape[0]),),
				                              dtype=np.float32)
			else:
				assert 0, 'Simulation type error'
		else:
			assert 0, 'No define state type error'

		self.action_space = spaces.Box(self.agent_params['minmax_value'][0], self.agent_params['minmax_value'][1],
		                               shape=(len(self.agent_params['variables_q0']),), dtype=np.float32)
		self.seed()
		self.viewer = None


	def step(self,actions: np.ndarray):
		self.trajectory_start_time = time()
		self.num_trajectory += 1
		if actions is None:
			print("carefully, no action given; by default, no jet!")

		self.action_time=0.025
		if self.vortex_shedding<0.01:
			self.vortex_shedding=0.01

		self.actions_sequence = np.append(self.actions_sequence, actions*2)
		if self.num_trajectory < 1.5:
			self.start_time_float = np.around(float(self.cfd_init_time_str),decimals=self.decimal)
		else:
			self.start_time_float = self.end_time_float
		self.end_time_float = np.around(self.start_time_float + self.action_time, decimals=self.decimal)

		#action
		self.control_matrix_gammaDMDc=self.control_matrix_gammaDMDc.flatten()
		self.control_matrix_gammaDMDc = np.delete(self.control_matrix_gammaDMDc, 0, axis=0)
		self.control_matrix_gammaDMDc = np.append(self.control_matrix_gammaDMDc, actions)

		self.action_time=self.delete_extra_zero(self.action_time)
		if self.num_trajectory < 1.5:
			self.start_actions = [0]
			self.end_actions = [self.actions_sequence[0]]
			action_tocsv_list = [[self.start_time_float, 0, 0, 0],[self.end_time_float, 0, self.actions_sequence[-1], 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet.csv', index=False, header=False)

		else:
			self.start_actions = [self.actions_sequence[-2]]
			self.end_actions = [self.actions_sequence[-1]]
			action_tocsv_list = [[self.start_time_float, 0, self.actions_sequence[-2], 0],[self.end_time_float, 0, self.actions_sequence[-1], 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet.csv', index=False, header=False)

		simulation_start_time = time()
		cfd.run(self.num_trajectory,
			self.foam_root_path,
			self.foam_params, self.action_time, self.agent_params['purgeWrite_numbers'],
			self.action_time,
			self.agent_params['deltaT'],
			self.start_time_float, self.end_time_float
		)
		simulation_end_time = time()

		if self.num_trajectory > 1.5:
			self.start_time_float = format(self.end_time_float - self.action_time, '.3f')  # 更改
		self.probe_velocity_df = utils.read_foam_file(
			self.foam_root_path + f'/postProcessing/probes/{self.start_time_float}/U',
			dimension=self.foam_params['num_dimension']
		)

		self.probe_pressure_df = utils.read_foam_file(
			self.foam_root_path + f'/postProcessing/probes/{self.start_time_float}/p',
			dimension=self.foam_params['num_dimension']
		)

		self.force_Coeffs_df = utils.read_foam_file(
			self.foam_root_path + f'/postProcessing/forceCoeffsIncompressible/{self.start_time_float}/forceCoeffs.dat'
		)

		if self.num_trajectory < 1.5:
			self.history_force_Coeffs_df = pd.concat(
				[self.history_force_Coeffs_df, self.force_Coeffs_df[:]],
				names=["Time", "Cm", "Cd", "Cl", "Cl(f)", "Cl(r)"]).reset_index(drop=True)
			self.history_force_Coeffs_df_stepnumber=self.force_Coeffs_df[:].shape[0]
		else:
			self.history_force_Coeffs_df = pd.concat(
				[self.history_force_Coeffs_df, self.force_Coeffs_df[1:]],names=[ "Time","Cm", "Cd", "Cl", "Cl(f)", "Cl(r)",]).reset_index(drop=True)
			self.history_force_Coeffs_df_stepnumber=self.force_Coeffs_df[:].shape[0]

		if self.state_params['type'] == 'pressure':
			next_state = self.probe_pressure_df.iloc[-1, 1:].to_numpy()
		elif self.state_params['tratype'] == 'velocity':
			next_state = self.probe_velocity_df.iloc[-1, 1:].to_numpy()
		else:
			next_state = False
			assert next_state, 'No define state type'
		next_state_record = next_state

		#DF-DRL
		if self.num_trajectory < 1.5:
			self.pressure_DMD_initial_snapshot = np.delete(self.pressure_DMD_initial_snapshot_init, 0, axis=1)
			self.pressure_DMD_initial_snapshot = np.hstack((self.pressure_DMD_initial_snapshot,
														next_state.reshape(
															int(self.state_params['probe_info'].shape[0]), 1)))
			self.pressure_DMD_initial_snapshot_step = self.pressure_DMD_initial_snapshot
		else:
			self.pressure_DMD_initial_snapshot = np.delete(self.pressure_DMD_initial_snapshot_step, 0, axis=1)
			self.pressure_DMD_initial_snapshot = np.hstack((self.pressure_DMD_initial_snapshot,
																 next_state.reshape(
																	 int(self.state_params['probe_info'].shape[0]),
																	 1)))
			self.pressure_DMD_initial_snapshot_step = self.pressure_DMD_initial_snapshot

		scaler = StandardScaler()
		self.pressure_DMD_initial_snapshot = scaler.fit_transform(self.pressure_DMD_initial_snapshot.T)
		self.control_matrix_gammaDMDc = self.control_matrix_gammaDMDc.reshape((30, 1))
		self.pressure_DMD_initial_snapshot = np.hstack((self.pressure_DMD_initial_snapshot, self.control_matrix_gammaDMDc))
		next_state = self.pressure_DMD_initial_snapshot.T

		self.state_data = np.append(self.state_data, next_state)

		cd_mean,cl_mean,cd_std,cl_std=self.reward_function()
		reward = -cd_mean- cl_mean
		print(self.num_trajectory, self.start_actions, self.end_actions,reward,self.action_time,self.vortex_shedding)
		self.trajectory_reward = np.append(self.trajectory_reward, reward)
		self.episode_reward += reward

		terminal = False

		self.trajectory_end_time = time()
		if self.num_trajectory > 1.5:
			self.start_time_float = np.around(self.end_time_float - self.action_time, decimals=self.decimal)

		self.exec_info = {
			'episode': self.num_episode,
			'trajectory': self.num_trajectory,
			'start_time_float': self.start_time_float,
			'end_time_float': self.end_time_float,
			'timestampStart': self.trajectory_start_time,
			'action_time':self.action_time,
			'vortex_shedding': self.vortex_shedding,
			'timestampEnd': self.trajectory_end_time,
			'current_trajectory_reward': reward,
			'episode_reward': self.episode_reward,
			'actions': actions,
			'cfd_running_time': simulation_end_time - simulation_start_time,
			'number_cfd_timestep': int(np.around((self.end_time_float - self.start_time_float) / self.foam_params['delta_t'])),
			'envName': self.foam_root_path.split('/')[-1],
			'current_state': self.state_data[-2],
			'next_state': next_state,
			'next_state_record': next_state_record,
		}
		self.info_list.append(self.exec_info)

		return next_state, reward, terminal, \
			   {'vortex_shedding_cd_mean':cd_mean, 'vortex_shedding_cl_mean':cl_mean,'vortex_shedding_cd_std':cd_std, 'vortex_shedding_cl_std':cl_std,'action':self.actions_sequence[-1], }

	@abstractmethod
	def reward_function(self):
		action_time = self.action_time
		vortex_shedding_period = self.vortex_shedding
		drug_coeffs_sliding_average = self.force_coeffs_sliding_average(action_time)[0]
		lift_coeffs_sliding_average = self.force_coeffs_sliding_average(action_time)[1]
		drug_coeffs_sliding_std = self.force_coeffs_sliding_std(vortex_shedding_period)[0]
		lift_coeffs_sliding_std = self.force_coeffs_sliding_std(vortex_shedding_period)[1]
		print(self.agent_params['cd_0'] - drug_coeffs_sliding_average,  lift_coeffs_sliding_std)

		return drug_coeffs_sliding_average,  np.abs(lift_coeffs_sliding_average), drug_coeffs_sliding_std,lift_coeffs_sliding_std

	def reset(self):
		if self.num_episode < 0.5:
			if self.foam_params['verbose']:
				if os.path.exists('record'):
					shutil.rmtree('record')
			os.makedirs(self.foam_root_path + '/record')
		else:
			self.episode_reward_sequence.append(self.episode_reward)
			pd.DataFrame(
				self.episode_reward_sequence
			).to_csv(self.foam_root_path + '/record/total_reward.csv', index=False, header=False)
			if self.episode_reward_sequence[-1] == np.max(self.episode_reward_sequence):
				pd.DataFrame(
					self.actions_sequence
				).to_csv(self.foam_root_path + '/record/best_actions.csv', index=False, header=False)
				pd.DataFrame(
					self.history_force_Coeffs_df
				).to_csv(self.foam_root_path + '/record/best_history_force_Coeffs_df.csv', index=False, header=False)
				with open(self.foam_root_path + '/record/info.txt', 'w') as f:
					f.write(f'Current number of best reward episode is {self.num_episode}')

		if self.num_episode == 1:
			self.all_episode_actions = pd.DataFrame(self.actions_sequence)
			self.all_episode_trajectory_reward = pd.DataFrame(self.trajectory_reward)
		else:
			self.all_episode_actions[self.num_episode - 1] = pd.DataFrame(self.actions_sequence)
			self.all_episode_trajectory_reward[self.num_episode - 1] = pd.DataFrame(self.trajectory_reward)

		self.all_episode_actions.to_csv(
			self.foam_root_path + '/record/all_episode_actions.csv', index=False, header=False
		)
		self.all_episode_trajectory_reward.to_csv(
			self.foam_root_path + '/record/all_episode_trajectory_reward.csv', index=False, header=False
		)
		self.history_force_Coeffs_df.to_csv(
			self.foam_root_path + f'/record/history_force_Coeffs_df_{self.num_episode}.csv', index=False, header=False
		)
		self.info_list = pd.DataFrame(self.info_list)
		self.info_list.to_csv(
			self.foam_root_path + f'/record/info_list_{self.num_episode}.csv'
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
		# action
		self.control_matrix_gammaDMDc = np.random.uniform(0, 0.01, 30)

		scaler = StandardScaler()
		self.pressure_DMD_initial_snapshot = scaler.fit_transform(self.pressure_DMD_initial_snapshot_init.T)
		self.control_matrix_gammaDMDc = self.control_matrix_gammaDMDc.reshape((30, 1))
		self.pressure_DMD_initial_snapshot = np.hstack(
			(self.pressure_DMD_initial_snapshot, self.control_matrix_gammaDMDc))
		self.dmdc_state = self.pressure_DMD_initial_snapshot.T  # 转置
		init_state=self.dmdc_state

		self.state_data = np.append(self.state_data, init_state)

		return init_state

	def close(self):
		super().close()

	def force_coeffs_sliding_average(self, sliding_time_interval):
		sampling_num = int(sliding_time_interval / self.foam_params['delta_t'])
		self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 2]=\
			signal.savgol_filter(self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 2], int(self.history_force_Coeffs_df_stepnumber/2), 0)
		self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 3] = \
			signal.savgol_filter(self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 3], int(self.history_force_Coeffs_df_stepnumber/2),0)
		sliding_average_cd = np.mean(self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 2])
		sliding_average_cl = np.mean(self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 3])
		return sliding_average_cd, sliding_average_cl

	def force_coeffs_sliding_std(self, sliding_time_interval):
		sampling_num = int(self.history_force_Coeffs_df_stepnumber / self.agent_params['action_discount'])
		if self.history_force_Coeffs_df.shape[0] <= sampling_num:
			sliding_average_cd_std = np.std(
				self.history_force_Coeffs_df.iloc[:, 2])
			sliding_average_cl_std = np.std(
				self.history_force_Coeffs_df.iloc[:, 3])
		else:
			sliding_average_cd_std = np.std(
				self.history_force_Coeffs_df.iloc[-sampling_num:, 2])
			sliding_average_cl_std = np.std(
				self.history_force_Coeffs_df.iloc[-sampling_num:, 3])
		return sliding_average_cd_std, sliding_average_cl_std

	def delete_extra_zero(self,n):
		if isinstance(n, int):
			return n
		if isinstance(n, float):
			n = str(n).rstrip('0')
			n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
			return n

	def seed(self, seed=0):
		self.rng = np.random.RandomState(seed)
		return [seed]

	def render(self, mode="human"):

		pass





