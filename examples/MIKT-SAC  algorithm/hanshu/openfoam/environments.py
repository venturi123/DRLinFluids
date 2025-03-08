from typing import Any, Callable, List, Optional, Tuple, Union

import gym
import numpy as np
import os
import re
import shutil
from time import time
from scipy import signal
import numpy as np
import pandas as pd
import random
from hanshu.openfoam import cfd, utils
from gym import spaces, logger
from gym.utils import seeding
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
		# 自动读取probes文件并提取信息
		self.state_params['probe_info'] = utils.read_foam_file(
			'/'.join([foam_root_path, 'system', 'probes'])
		)
		# 记录每个trajectory的一系列变量并传递给plotly
		self.dashboard_data = {}
		# 记录每个Trajectory开始时间戳
		self.trajectory_start_time = 0
		# 记录每个Trajectory结束时间戳
		self.trajectory_end_time = 0
		# 记录episode次数
		self.num_episode = 165
		# 记录所有step信息
		self.info_list = []
		# 记录每个episode的reward
		self.episode_reward_sequence = []
		# 用以传递execute()过程中产生的数据
		self.exec_info = {}
		# 初始化trajectory计数器
		self.num_trajectory = 0
		# 初始化每一步（trajectory）奖励
		self.trajectory_reward = np.array([])
		# 初始化记录所有episode的所有trajectory奖励
		self.all_episode_trajectory_reward = pd.DataFrame()
		# 记录当前Episode所有Trajectory的State列表
		self.state_data = np.array([])
		# 初始化累计奖励
		self.episode_reward = -1000
		# 当前步实际输入到OpenFOAM中的限制后action
		self.decorated_actions = np.array([])
		# 初始化episode中的action序列
		self.actions_sequence = np.array([])
		self.actions_sequence_jet1 = np.array([])
		self.actions_sequence_jet2 = np.array([])
		self.actions_sequence_jet3 = np.array([])
		self.actions_sequence_jet4 = np.array([])
		# 初始化实际action动作序列（考虑到agent输出action与输入cfd中的边界条件间多了一层函数）
		# self.decorated_actions_sequence = np.array([])
		# 初始化steps中的action序列
		self.start_actions_jet1 = 0
		self.end_actions_jet1 = 0
		self.start_actions_jet2 = 0
		self.end_actions_jet2 = 0
		self.start_actions_jet3 = 0
		self.end_actions_jet3 = 0
		self.start_actions_jet4 = 0
		self.end_actions_jet4 = 0
		self.start_actions = 0
		self.end_actions = 0
		self.actions_mean = 0    #所有actions的均值
		# 记录steps中的actions
		self.single_step_actions = np.array([])
		# 初始化一个pandas对象，用以记录所有episode的所有action
		self.all_episode_actions = pd.DataFrame()
		self.all_episode_actions_jet1 = pd.DataFrame()
		self.all_episode_actions_jet2 = pd.DataFrame()
		self.all_episode_actions_jet3 = pd.DataFrame()
		self.all_episode_actions_jet4 = pd.DataFrame()
		# 初始化一个pandas对象，用以记录所有episode的所有实际输出的action
		self.all_episode_decorated_actions = pd.DataFrame()
		# 初始化一个pandas对象，用以记录所有episode的所有实际输出的action
		self.all_episode_single_step_actions = pd.DataFrame()
		# 初始化velocity(state)文件
		self.probe_velocity_df = pd.DataFrame()
		# 初始化pressure(state)文件
		self.probe_pressure_df = pd.DataFrame()
		# 初始化当前时间步内forces(reward)
		self.force_df = pd.DataFrame()
		# 初始化当前时间步内力系数forceCoeffs(reward)
		self.force_Coeffs_df = pd.DataFrame()
		# 初始化全周期力forces(reward)
		self.history_force_df = pd.DataFrame()
		# 初始化全周期力系数forceCoeffs(reward)
		self.initial_force_Coeffs_df = pd.DataFrame()
		self.history_force_Coeffs_df = pd.DataFrame()
		self.history_force_Coeffs_df_alltime = pd.DataFrame()
		self.history_force_Coeffs_df_stepnumber=0
		self.start_time_float=0
		self.end_time_float=0
		self.start_time_filename="1"
		self.start_time_path="1"
		self.terminal=False
		#action
		self.control_matrix1 = np.array([])
		self.control_matrix2 = np.array([])
		self.control_matrix3 = np.array([])
		self.control_matrix4 = np.array([])

		# 读取输入的初始化流场时刻，调整其格式
		self.cfd_init_time_str = str(float(foam_params['cfd_init_time'])).rstrip('0').rstrip('.')
		# 保存INTERVENTION_T和INIT_START_TIME的小数位长度最大值，后续在读取文件时，会根据该变量，自动截取有效位数，
		# 避免在进制转换时出现误差而导致不能正确读取文件
		self.decimal = int(np.max([
			len(str(agent_params['interaction_period']).split('.')[-1]),
			len(str(foam_params['cfd_init_time']).split('.')[-1])
		]))
		self.pressure_DMD_initial_snapshot=np.array([])
		self.control_matrix_gammaDMDc=np.array([])

		if server:
			# if foam_params['cfd_init_time'] > 0:
			#     # 初始化一个流场，不然初始state为空，无法进行学习
			# 初始化jet.csv
			action_tocsv_list = [[0, 0, 0, 0],
				[self.foam_params['cfd_init_time'], 0, 0, 0]]
			pd.DataFrame(
					action_tocsv_list
				).to_csv(self.foam_root_path + '/system/jet1.csv', index=False, header=False)
			pd.DataFrame(
					action_tocsv_list
				).to_csv(self.foam_root_path + '/system/jet2.csv', index=False, header=False)
			pd.DataFrame(
					action_tocsv_list
				).to_csv(self.foam_root_path + '/system/jet3.csv', index=False, header=False)
			pd.DataFrame(
				action_tocsv_list
			).to_csv(self.foam_root_path + '/system/jet4.csv', index=False, header=False)
			pd.DataFrame(
				action_tocsv_list
			).to_csv(self.foam_root_path + '/system/jet5.csv', index=False, header=False)
			pd.DataFrame(
				action_tocsv_list
			).to_csv(self.foam_root_path + '/system/jet6.csv', index=False, header=False)
			pd.DataFrame(
				action_tocsv_list
			).to_csv(self.foam_root_path + '/system/jet7.csv', index=False, header=False)
			pd.DataFrame(
				action_tocsv_list
			).to_csv(self.foam_root_path + '/system/jet8.csv', index=False, header=False)
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

			# 标准化所需信息
			self.initial_force_Coeffs_df = utils.read_foam_file(
				foam_root_path + f'/forceCoeffsIncompressible/0.000/forceCoeffs.dat'
			)
			self.pressure_DMD_initial = utils.read_foam_file(
				foam_root_path + f'/probes/0.000/p',
				dimension=self.foam_params['num_dimension']
			)
			self.pressure_DMD_initial = self.pressure_DMD_initial.iloc[1:, 1:].to_numpy().T
			#0.39
			self.pressure_DMD_initial_snapshot=self.pressure_DMD_initial[:,(int(self.agent_params['interaction_period'] / self.foam_params['delta_t'])-1)
																		   :(int(self.agent_params['interaction_period'] / self.foam_params['delta_t'])*30):
																		   (int(self.agent_params['interaction_period'] / self.foam_params['delta_t']))]
			# 保留初始状态
			self.pressure_DMD_initial_snapshot_init=self.pressure_DMD_initial_snapshot

		#加入action维度
		self.dmd_state=np.zeros((int(self.state_params['probe_info'].shape[0]),30), dtype=int)
		# 将pressure作为训练参数
		if self.state_params['type'] == 'pressure':
			self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
			                              shape=(self.dmd_state.shape), dtype=np.float32)
		# velocity作为训练参数
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

		# action_space
		self.action_space = spaces.Box(self.agent_params['minmax_value'][0], self.agent_params['minmax_value'][1],
		                               shape=(len(self.agent_params['variables_q0'])*4,), dtype=np.float32)
		self.seed()
		self.viewer = None

	# 接受一个动作并返回一个元组,运行环境动态的一个timestep,结束 时重置此环境的状态
	def execute(self,actions: np.ndarray):
		self.trajectory_start_time = time()
		# 记录trajectory回合数
		self.num_trajectory += 1
		# assert 0, f'{actions}\n{type(actions)}'
		if actions is None:
			print("carefully, no action given; by default, no jet!")

		self.action_time=0.005
		self.vortex_shedding=0.06

		self.actions_mean = np.mean(sum(actions))
		self.actions_sequence_jet1 = np.append(self.actions_sequence_jet1, (actions[0]) * 3)
		self.actions_sequence_jet2 = np.append(self.actions_sequence_jet2, (actions[1]) * 3)
		self.actions_sequence_jet3 = np.append(self.actions_sequence_jet3, (actions[2]) * 3)
		self.actions_sequence_jet4 = np.append(self.actions_sequence_jet4, (actions[3]) * 3)

		#action
		# self.control_matrix1=self.control_matrix1.flatten()
		# self.control_matrix1 = np.delete(self.control_matrix1, 0, axis=0)
		# self.control_matrix1 = np.append(self.control_matrix1, (actions[0]))
		# self.control_matrix2 = self.control_matrix2.flatten()
		# self.control_matrix2 = np.delete(self.control_matrix2, 0, axis=0)
		# self.control_matrix2 = np.append(self.control_matrix2, (actions[1]))
		# self.control_matrix3 = self.control_matrix3.flatten()
		# self.control_matrix3 = np.delete(self.control_matrix3, 0, axis=0)
		# self.control_matrix3 = np.append(self.control_matrix3, (actions[2]))
		# self.control_matrix4 = self.control_matrix4.flatten()
		# self.control_matrix4 = np.delete(self.control_matrix4, 0, axis=0)
		# self.control_matrix4 = np.append(self.control_matrix4,(actions[3]))
		

		if self.num_trajectory < 1.5:
			self.start_time_float =0
			self.end_time_float=np.around( self.action_time, decimals=self.decimal)
		else:
			self.start_time_float = self.end_time_float
			self.end_time_float = np.around(self.start_time_float + self.action_time, decimals=self.decimal)


		if self.num_trajectory < 1.5:
			self.start_actions_jet1 = [0]
			self.start_actions_jet2 = [0]
			self.start_actions_jet3 = [0]
			self.start_actions_jet4 = [0]
			self.end_actions_jet1 = [self.actions_sequence_jet1[0]]
			self.end_actions_jet2 = [self.actions_sequence_jet2[0]]
			self.end_actions_jet3 = [self.actions_sequence_jet3[0]]
			self.end_actions_jet4 = [self.actions_sequence_jet4[0]]
			action_tocsv_list = [[self.start_time_float, 0, 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'],
								  self.actions_sequence_jet4[-1], 0, 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet1.csv', index=False, header=False)
			action_tocsv_list = [[self.start_time_float, 0, 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'], 0,
								  self.actions_sequence_jet4[-1], 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet8.csv', index=False, header=False)

			action_tocsv_list = [[self.start_time_float, 0, 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'],
								  self.actions_sequence_jet1[-1], 0, 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet2.csv', index=False, header=False)
			action_tocsv_list = [[self.start_time_float, 0, 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'], 0,
								  -self.actions_sequence_jet1[-1], 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet3.csv', index=False, header=False)

			action_tocsv_list = [[self.start_time_float, 0, 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'], 0,
								  -self.actions_sequence_jet2[-1], 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet4.csv', index=False, header=False)
			action_tocsv_list = [[self.start_time_float, 0, 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'],
								  -self.actions_sequence_jet2[-1], 0, 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet5.csv', index=False, header=False)

			action_tocsv_list = [[self.start_time_float, 0, 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'],
								  -self.actions_sequence_jet3[-1], 0, 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet6.csv', index=False, header=False)
			action_tocsv_list = [[self.start_time_float, 0, 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'], 0,
								  self.actions_sequence_jet3[-1], 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet7.csv', index=False, header=False)

		else:
			self.start_actions_jet1 = [self.actions_sequence_jet1[-2]]
			self.end_actions_jet1 = [self.actions_sequence_jet1[-1]]
			self.start_actions_jet2 = [self.actions_sequence_jet2[-2]]
			self.end_actions_jet2 = [self.actions_sequence_jet2[-1]]
			self.start_actions_jet3 = [self.actions_sequence_jet3[-2]]
			self.end_actions_jet3 = [self.actions_sequence_jet3[-1]]
			self.start_actions_jet4 = [self.actions_sequence_jet4[-2]]
			self.end_actions_jet4 = [self.actions_sequence_jet4[-1]]
			action_tocsv_list = [[self.start_time_float, self.actions_sequence_jet4[-2], 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'],
								  self.actions_sequence_jet4[-1], 0, 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet1.csv', index=False, header=False)
			action_tocsv_list = [[self.start_time_float, 0, self.actions_sequence_jet4[-2], 0],
								 [self.start_time_float + self.agent_params['interaction_period'], 0,
								  self.actions_sequence_jet4[-1], 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet8.csv', index=False, header=False)

			action_tocsv_list = [[self.start_time_float, self.actions_sequence_jet1[-2], 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'],
								  self.actions_sequence_jet1[-1], 0, 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet2.csv', index=False, header=False)
			action_tocsv_list = [[self.start_time_float, 0, -self.actions_sequence_jet1[-2], 0],
								 [self.start_time_float + self.agent_params['interaction_period'], 0,
								  -self.actions_sequence_jet1[-1], 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet3.csv', index=False, header=False)

			action_tocsv_list = [[self.start_time_float, 0, -self.actions_sequence_jet2[-2], 0],
								 [self.start_time_float + self.agent_params['interaction_period'], 0,
								  -self.actions_sequence_jet2[-1], 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet4.csv', index=False, header=False)
			action_tocsv_list = [[self.start_time_float, -self.actions_sequence_jet2[-2], 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'],
								  -self.actions_sequence_jet2[-1], 0, 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet5.csv', index=False, header=False)

			action_tocsv_list = [[self.start_time_float, -self.actions_sequence_jet3[-2], 0, 0],
								 [self.start_time_float + self.agent_params['interaction_period'],
								  -self.actions_sequence_jet3[-1], 0, 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet6.csv', index=False, header=False)
			action_tocsv_list = [[self.start_time_float, 0, self.actions_sequence_jet3[-2], 0],
								 [self.start_time_float + self.agent_params['interaction_period'], 0,
								  self.actions_sequence_jet3[-1], 0]]
			pd.DataFrame(action_tocsv_list).to_csv(self.foam_root_path + '/system/jet7.csv', index=False, header=False)

		# self.start_time_float = np.around(
		# 	float(self.cfd_init_time_str) + (self.num_trajectory - 1) * self.agent_params['interaction_period'],
		# 	decimals=self.decimal
		# )
		# self.end_time_float = np.around(self.start_time_float + self.agent_params['interaction_period'], decimals=self.decimal)

		# 找到当前最新时间文件夹，作为startTime，用以指定actions写入文件夹路径
		# self.start_time_filename, self.start_time_path = utils.get_current_time_path(self.foam_root_path)

		simulation_start_time = time()
		# print("test",self.action_time)
		cfd.run(self.num_trajectory,
			self.foam_root_path,
			self.foam_params, self.action_time, self.agent_params['purgeWrite_numbers'],
			# self.agent_params['writeInterval'],
			self.action_time,
			self.agent_params['deltaT'],
			self.start_time_float, self.end_time_float
		)
		simulation_end_time = time()

		#start_time_float转化为str  增加零位
		if self.num_trajectory < 1.5:
			self.start_time_float = format(0.000, '.3f')  # 更改
		else:
			self.start_time_float = format(self.end_time_float - self.action_time, '.3f')  # 更改

		# 读取velocity(state)文件
		if self.num_trajectory > 1.5:
			self.start_time_float = format(self.end_time_float - self.action_time, '.3f')  # 更改
		# 读取velocity(state)文件
		self.probe_velocity_df = utils.read_foam_file(
			self.foam_root_path + f'/postProcessing/probes/{self.start_time_float}/U',
			dimension=self.foam_params['num_dimension']
		)

		# 读取pressure文件(state)
		self.probe_pressure_df = utils.read_foam_file(
			self.foam_root_path + f'/postProcessing/probes/{self.start_time_float}/p',
			dimension=self.foam_params['num_dimension']
		)

		# 读取力系数forceCoeffs.dat文件(reward)
		self.force_Coeffs_df = utils.read_foam_file(
			self.foam_root_path + f'/postProcessing/forceCoeffsIncompressible/{self.start_time_float}/forceCoeffs.dat'
		)

		# 连接当前trajectory之前的所有全周期历史力和力系数数据
		if self.num_trajectory < 1.5:
			# self.history_force_df = self.force_df
			self.history_force_Coeffs_df = pd.concat(
				[self.history_force_Coeffs_df, self.force_Coeffs_df[:]],
				names=["Time", "Cm", "Cd", "Cl", "Cl(f)", "Cl(r)"]).reset_index(drop=True)
			self.history_force_Coeffs_df_stepnumber = self.force_Coeffs_df[:].shape[0]
		else:
			self.history_force_Coeffs_df = pd.concat(
				[self.history_force_Coeffs_df, self.force_Coeffs_df[1:]],names=[ "Time","Cm", "Cd", "Cl", "Cl(f)", "Cl(r)",]).reset_index(drop=True)
			self.history_force_Coeffs_df_stepnumber = self.force_Coeffs_df[:].shape[0]


		# 将结果文件的最后一行作为下一个state状态
		next_state=[]
		if self.state_params['type'] == 'pressure':
			next_state = self.probe_pressure_df.iloc[-1, 1:].to_numpy()
		elif self.state_params['tratype'] == 'velocity':
			next_state = self.probe_velocity_df.iloc[-1, 1:].to_numpy()
		else:
			next_state = False
			assert next_state, 'No define state type'
		next_state_record = next_state

		if self.num_trajectory < 1.5:
			# 进行转置，删除state最开始那一列,变为(6.29)
			self.pressure_DMD_initial_snapshot = np.delete(self.pressure_DMD_initial_snapshot_init, 0, axis=1)
			# 变为(20.30)
			self.pressure_DMD_initial_snapshot = np.hstack((self.pressure_DMD_initial_snapshot,
														next_state.reshape(
															int(self.state_params['probe_info'].shape[0]), 1)))
			#记录真实值
			self.pressure_DMD_initial_snapshot_step = self.pressure_DMD_initial_snapshot
		else:
			# 进行转置，删除state最开始那一列,变为(6.29)
			self.pressure_DMD_initial_snapshot = np.delete(self.pressure_DMD_initial_snapshot_step, 0, axis=1)
			# 变为(20.30)
			self.pressure_DMD_initial_snapshot = np.hstack((self.pressure_DMD_initial_snapshot,
																 next_state.reshape(
																	 int(self.state_params['probe_info'].shape[0]),
																	 1)))
			#保存
			self.pressure_DMD_initial_snapshot_step = self.pressure_DMD_initial_snapshot

		#只对state做规整化,但只对列有效果
		scaler = StandardScaler()
		#变为(30.7)
		self.pressure_DMD_initial_snapshot = scaler.fit_transform(self.pressure_DMD_initial_snapshot.T)
		#加入action
		# self.control_matrix1 = self.control_matrix1.reshape((30, 1))
		# self.control_matrix2 = self.control_matrix2.reshape((30, 1))
		# self.control_matrix3 = self.control_matrix3.reshape((30, 1))
		# self.control_matrix4 = self.control_matrix4.reshape((30, 1))
		# self.pressure_DMD_initial_snapshot = np.hstack((self.pressure_DMD_initial_snapshot, self.control_matrix1))
		# self.pressure_DMD_initial_snapshot = np.hstack((self.pressure_DMD_initial_snapshot, self.control_matrix2))
		# self.pressure_DMD_initial_snapshot = np.hstack((self.pressure_DMD_initial_snapshot, self.control_matrix3))
		# self.pressure_DMD_initial_snapshot = np.hstack((self.pressure_DMD_initial_snapshot, self.control_matrix4))
		# (30,7)变为(7,30)
		next_state = self.pressure_DMD_initial_snapshot.T

		self.state_data = np.append(self.state_data, next_state)

		cd_mean,cl_mean,cd_std,cl_std=self.reward_function()
		reward = -cd_mean- cl_mean
		print(self.num_trajectory, self.end_actions_jet1, self.end_actions_jet2, self.end_actions_jet3,
			  self.end_actions_jet4, reward)
		# 记录每一个trajectory和episode的奖励值
		self.trajectory_reward = np.append(self.trajectory_reward, reward)
		self.episode_reward += reward

		# TODO 中止条件需要认真考虑，可以考虑到total_reward低于某个值的时候停止
		if self.num_trajectory == 80:    #跑不了太多步
			self.terminal = True

		self.trajectory_end_time = time()
		if self.num_trajectory < 1.5:
			self.start_time_float = 0  # 更改
		else:
			self.start_time_float = np.around(self.end_time_float - self.action_time, decimals=self.decimal)

		# 用以传递execute()过程中产生的数据
		self.exec_info = {
			'episode': self.num_episode,  # 写成文件名，self.num_episode.csv
			'trajectory': self.num_trajectory,
			'start_time_float': self.start_time_float,
			'end_time_float': self.end_time_float,
			'timestampStart': self.trajectory_start_time,
			'action_time':self.action_time,
			'vortex_shedding': self.vortex_shedding,
			'timestampEnd': self.trajectory_end_time,
			'current_trajectory_reward': reward,
			'episode_reward': self.episode_reward,
			'actions': actions,  # 数组
			'cfd_running_time': simulation_end_time - simulation_start_time,
			'number_cfd_timestep': int(np.around((self.end_time_float - self.start_time_float) / self.foam_params['delta_t'])),
			'envName': self.foam_root_path.split('/')[-1],  # str
			'current_state': self.state_data[-2],
			'next_state': next_state,
			'next_state_record': next_state_record,
		}
		self.info_list.append(self.exec_info)

		return next_state, reward, self.terminal, \
			   {'vortex_shedding_cd_mean':cd_mean, 'vortex_shedding_cl_mean':cl_mean,'vortex_shedding_cd_std':cd_std,
				'vortex_shedding_cl_std':cl_std,'action1':self.actions_sequence_jet1[-1],'action2':self.actions_sequence_jet2[-1],
				'action3':self.actions_sequence_jet3[-1],'action4':self.actions_sequence_jet4[-1],}

	# 定义reward，并制定为抽象方法（这也是为什么要继承ABC类的原因），这意味着在实例化OpenFoam类时，必须覆盖reward_function函数
	# @abstractmethod
	def reward_function(self):
		action_time = self.action_time
		vortex_shedding_period = self.vortex_shedding
		drug_coeffs_sliding_average = self.force_coeffs_sliding_average(action_time)[0]
		lift_coeffs_sliding_average = self.force_coeffs_sliding_average(action_time)[1]
		drug_coeffs_sliding_std = self.force_coeffs_sliding_std(vortex_shedding_period)[0]
		lift_coeffs_sliding_std = self.force_coeffs_sliding_std(vortex_shedding_period)[1]
		print( drug_coeffs_sliding_average,  lift_coeffs_sliding_average)

		return drug_coeffs_sliding_average,  np.abs(lift_coeffs_sliding_average), drug_coeffs_sliding_std,lift_coeffs_sliding_std

	# 此处的reset是用于episode之间的重置，不是每个trajectory的重置，注意会在initial之后进行一次reset再进行execute
	# 将环境重置为初始状态，并返回初始状态观察。
	def reset(self):
		# 保存最高奖励episode
		if self.num_episode < 0.5:
			if os.path.exists(self.foam_root_path + '/record'):
				# 如果⽬标路径存在原⽂件夹的话就先删除
				shutil.rmtree(self.foam_root_path + '/record')
			os.makedirs(self.foam_root_path + '/record')
		# pass
		else:
			# 提取整个episode中的最优action，跳过第一次初始化过程
			self.episode_reward_sequence.append(self.episode_reward)
			# 如果最近一次episode获得的reward为目前的最大值，则将这一步所记录的self.actions_record当做最优action
			pd.DataFrame(
				self.episode_reward_sequence
			).to_csv(self.foam_root_path + '/record/total_reward.csv', index=False, header=False)
			if self.episode_reward_sequence[-1] == np.max(self.episode_reward_sequence):
				pd.DataFrame(
					self.actions_sequence_jet1
				).to_csv(self.foam_root_path + '/record/best_actions_jet1.csv', index=False, header=False)
				pd.DataFrame(
					self.actions_sequence_jet2
				).to_csv(self.foam_root_path + '/record/best_actions_jet2.csv', index=False, header=False)
				pd.DataFrame(
					self.actions_sequence_jet3
				).to_csv(self.foam_root_path + '/record/best_actions_jet3.csv', index=False, header=False)
				pd.DataFrame(
					self.actions_sequence_jet4
				).to_csv(self.foam_root_path + '/record/best_actions_jet4.csv', index=False, header=False)
				# self.history_force_Coeffs_df[:, 3]=signal.savgol_filter(self.force_Coeffs_df.iloc[:, 3], 11, 3)
				pd.DataFrame(
					self.history_force_Coeffs_df
				).to_csv(self.foam_root_path + '/record/best_history_force_Coeffs_df.csv', index=False, header=False)
				with open(self.foam_root_path + '/record/info.txt', 'w') as f:
					f.write(f'Current number of best reward episode is {self.num_episode}')

		# 输出截止到当前trajectory的所有action
		if self.num_episode == 1:
			self.all_episode_actions_jet1 = pd.DataFrame(self.actions_sequence_jet1)
			self.all_episode_actions_jet2 = pd.DataFrame(self.actions_sequence_jet2)
			self.all_episode_actions_jet3 = pd.DataFrame(self.actions_sequence_jet3)
			self.all_episode_actions_jet4 = pd.DataFrame(self.actions_sequence_jet4)
			#self.all_episode_decorated_actions = pd.DataFrame(self.decorated_actions_sequence)
			self.all_episode_trajectory_reward = pd.DataFrame(self.trajectory_reward)
		# self.all_episode_single_step_actions = pd.DataFrame(self.single_step_actions)
		else:
			self.all_episode_actions_jet1[self.num_episode - 1] = pd.DataFrame(self.actions_sequence_jet1)
			self.all_episode_actions_jet2[self.num_episode - 1] = pd.DataFrame(self.actions_sequence_jet2)
			self.all_episode_actions_jet3[self.num_episode - 1] = pd.DataFrame(self.actions_sequence_jet3)
			self.all_episode_actions_jet4[self.num_episode - 1] = pd.DataFrame(self.actions_sequence_jet4)
			self.all_episode_trajectory_reward[self.num_episode - 1] = pd.DataFrame(self.trajectory_reward)

		# 保存agent直出actions和修饰后actions以及每一个trajectory的奖励值
		self.all_episode_actions_jet1.to_csv(
			self.foam_root_path + '/record/all_episode_actions_jet1.csv', index=False, header=False
		)
		self.all_episode_actions_jet2.to_csv(
			self.foam_root_path + '/record/all_episode_actions_jet2.csv', index=False, header=False
		)
		self.all_episode_actions_jet3.to_csv(
			self.foam_root_path + '/record/all_episode_actions_jet3.csv', index=False, header=False
		)
		self.all_episode_actions_jet4.to_csv(
			self.foam_root_path + '/record/all_episode_actions_jet4.csv', index=False, header=False
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
		# 更新episode
		self.num_episode += 1
		# 重置一个episode中的trajectory
		self.num_trajectory = 0
		# 重置累计reward
		self.episode_reward = 0
		# 重置episode中的action序列
		self.actions_sequence_jet1 = []
		self.actions_sequence_jet2 = []
		self.actions_sequence_jet3 = []
		self.actions_sequence_jet4 = []
		self.trajectory_reward = []
		# 重置实际action动作序列（考虑到agent输出action与输入cfd中的边界条件间多了一层函数）
		# self.decorated_actions_sequence = []
		# 重置历史力及力系数
		self.history_force_df = pd.DataFrame()
		self.history_force_Coeffs_df = pd.DataFrame()
		self.info_list = []
		self.terminal=False
		# TODO 流场需要初始化，删除所有文件，但不用重置控制字典，因为在execute()函数中已经包含了设定计算时间
		# 删除除0和初始流场文件夹以外所有时间文件夹以及后处理文件夹postProcessing
		# 返回当前工作目录上级目录的所有文件名称：os.listdir('/'.join(self.foam_root_path.split('/')[:-1]))
		for f_name in os.listdir(self.foam_root_path):
			if re.search(r'^\d+\.?\d*', f_name):
				if (f_name != '0') and (f_name != self.cfd_init_time_str):
					shutil.rmtree('/'.join([self.foam_root_path, f_name]))
			elif f_name == 'postProcessing':
				shutil.rmtree('/'.join([self.foam_root_path, f_name]))
			else:
				pass

		# self.control_matrix1 = np.random.uniform(0, 0.01, 30)
		# self.control_matrix2 = np.random.uniform(0, 0.01, 30)
		# self.control_matrix3 = np.random.uniform(0, 0.01, 30)
		# self.control_matrix4 = np.random.uniform(0, 0.01, 30)
		# （6.30）变为(30.6)
		scaler = StandardScaler()
		self.pressure_DMD_initial_snapshot = scaler.fit_transform(self.pressure_DMD_initial_snapshot_init.T)
		# 加入action
		# self.control_matrix1 = self.control_matrix1.reshape((30, 1))
		# self.control_matrix2 = self.control_matrix2.reshape((30, 1))
		# self.control_matrix3 = self.control_matrix3.reshape((30, 1))
		# self.control_matrix4 = self.control_matrix4.reshape((30, 1))
		# self.pressure_DMD_initial_snapshot = np.hstack(
		# 	(self.pressure_DMD_initial_snapshot, self.control_matrix1))
		# self.pressure_DMD_initial_snapshot = np.hstack(
		# 	(self.pressure_DMD_initial_snapshot, self.control_matrix2))
		# self.pressure_DMD_initial_snapshot = np.hstack(
		# 	(self.pressure_DMD_initial_snapshot, self.control_matrix3))
		# self.pressure_DMD_initial_snapshot = np.hstack(
		# 	(self.pressure_DMD_initial_snapshot, self.control_matrix4))
		# (30,7)变为(7,30)
		self.dmdc_state = self.pressure_DMD_initial_snapshot.T  # 转置

		init_state=self.dmdc_state

		# 将初始state放入state数组中
		self.state_data = np.append(self.state_data, init_state)

		return init_state

	# def max_episode_timesteps(self):
	# 	return super().max_episode_timesteps()

	def close(self):
		super().close()

	def force_coeffs_sliding_average(self, sliding_time_interval):
		# 计算采样时间点数
		sampling_num = int(sliding_time_interval / self.foam_params['delta_t'])
		self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 2] = signal.savgol_filter(
			self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 2], 1, 0)
		self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 3] = signal.savgol_filter(
			self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 3], 1, 0)
		sliding_average_cd = np.mean(self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 2])
		sliding_average_cl = np.mean(self.history_force_Coeffs_df.iloc[-self.history_force_Coeffs_df_stepnumber:, 3])
		return sliding_average_cd, sliding_average_cl

	def force_coeffs_sliding_std(self, sliding_time_interval):
		# 计算采样时间点数
		sampling_num = int(self.history_force_Coeffs_df_stepnumber / self.agent_params['action_discount'])
		# 计算一个旋涡脱落周期内的升力系数滑动平均值
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

	def fft(self):
		sampling_num = int(self.agent_params['interaction_period'] *20 / self.foam_params['delta_t'])
		#初始30个action_cl=0.51
		# x = list(self.history_force_Coeffs_df.iloc[-sampling_num:, 3])
		self.history_force_Coeffs_df_alltime = pd.concat(
			[self.initial_force_Coeffs_df[1:], self.history_force_Coeffs_df]
		).reset_index(drop=True)
		x = list(signal.savgol_filter(self.history_force_Coeffs_df_alltime.iloc[-sampling_num:, 3], 49,0))

		fs = 4000  # fs=1/dt   square
		ps_x = np.abs(np.fft.fft(x)) ** 2
		freqs_x = np.fft.fftfreq(self.history_force_Coeffs_df_alltime.iloc[-sampling_num:, 3].size, 1 / fs)
		ps_x = ps_x * freqs_x
		freqs_x = freqs_x * 0.0508 / 2  # 计算斯托罗哈数
		# freqs_x=freqs_x*0.1/1  #计算斯托罗哈数
		ymax1 = max(ps_x)
		xpos1 = list(ps_x).index(ymax1)
		xmax1 = freqs_x[xpos1]

		vortex_shedding_frequence = xmax1*2/0.0508
		vortex_shedding= np.around(
			1/vortex_shedding_frequence,
			decimals=self.decimal
		)
		action_time= np.around(
			1/vortex_shedding_frequence*self.agent_params['action_discount'],
			decimals=self.decimal
		)
		return vortex_shedding,action_time

	def delete_extra_zero(self,n):
		'''删除小数点后多余的0'''
		if isinstance(n, int):
			return n
		if isinstance(n, float):
			n = str(n).rstrip('0')  # 删除小数点后多余的0
			n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
			return n

	def seed(self, seed=0):
		self.rng = np.random.RandomState(seed)
		return [seed]

	# 渲染环境,渲染到当前显示或终端不返回任何内容
	def render(self, mode="human"):

		pass


