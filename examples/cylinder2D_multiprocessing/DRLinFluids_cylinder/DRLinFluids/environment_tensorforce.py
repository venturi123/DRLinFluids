# coding = UTF-8
import os
import re
import shutil
from abc import ABCMeta, abstractmethod
from time import time
from scipy import signal
import numpy as np
import pandas as pd
from DRLinFluids import cfd, utils
from tensorforce.environments import Environment


class OpenFoam_tensorforce(Environment, metaclass=ABCMeta):
    """The main DRLinFluids tensorforce class.
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
    from DRLinFluids.environments_tensorforce import OpenFoam_tensorforce

    note
    --------
    The main API methods that users of this class need to know are:
    - :meth:`states` - Define state_space.
    - :meth:`actions` - Define action_space.
    - :meth:`step` - Takes a step in the environment using an action returning the next observation, reward,
    if the environment terminated and more information.
    - :meth:`reward_function` - Define reward_funtion and calculate.
    - :meth:`reset` - Resets the environment to an initial state, returning the initial observation.
    """
    def __init__(self, foam_root_path, foam_params, agent_params, state_params,server=True):
        super().__init__()
        self.foam_root_path = foam_root_path
        self.foam_params = foam_params
        self.agent_params = agent_params
        self.state_params = state_params
        # Automatically read probes files and extract information
        self.state_params['probe_info'] = utils.read_foam_file(
            '/'.join([foam_root_path, 'system', 'probes'])
        )
        # Record a series of variables for each trajectory and pass to plotly
        self.dashboard_data = {}
        # Record each Trajectory start timestamp
        self.trajectory_start_time = 0
        # Record each Trajectory end timestamp
        self.trajectory_end_time = 0
        # Record the number of episodes
        self.num_episode = 0
        # Record all step information
        self.info_list=[]
        # Record the reward for each episode
        self.episode_reward_sequence = []
        # Used to pass the data generated during the execute() process
        self.exec_info = {}
        # Initialize the trajectory counter
        self.num_trajectory = 0
        # Initialize each step (trajectory) reward
        self.trajectory_reward = np.array([])
        # Initially record all trajectory rewards for all episodes
        self.all_episode_trajectory_reward = pd.DataFrame()
        # Record the State list of all Trajectory in the current Episode
        self.state_data = np.array([])
        # Initialize cumulative rewards
        self.episode_reward = 0
        # Initialize the action sequence in the episode
        self.actions_sequence = np.array([])
        # Action at the start of each step
        self.start_actions = 0
        # Action at the end of each step
        self.end_actions = 0
        # Record actions in steps
        self.single_step_actions = np.array([])
        # Initialize a pandas object to record all actions of all episodeson
        self.all_episode_actions = pd.DataFrame()
        # Initialize a pandas object to record all actual output actions of all episodes
        self.all_episode_decorated_actions = pd.DataFrame()
        # Initialize a pandas object to record all actual output actions of all episodes
        self.all_episode_single_step_actions = pd.DataFrame()
        # Initialize the velocity(state) file
        self.probe_velocity_df = pd.DataFrame()
        # Initialize the pressure(state) file
        self.probe_pressure_df = pd.DataFrame()
        # Initialize forces(reward) in the current time step
        self.force_df = pd.DataFrame()
        # Initialize the current time step internal force coefficient forceCoeffs(reward)
        self.force_Coeffs_df = pd.DataFrame()
        # Initialize full cycle force forces(reward)
        self.history_force_df = pd.DataFrame()
        # Initialize the full cycle force coefficient forceCoeffs(reward)
        self.history_force_Coeffs_df = pd.DataFrame()
        # Read the initial flow field moment of the input and adjust its format
        self.cfd_init_time_str = str(float(foam_params['cfd_init_time'])).rstrip('0').rstrip('.')
        # Avoid errors in hex conversion, resulting in inability to read the file correctly
        self.decimal = int(np.max([
                    len(str(agent_params['interaction_period']).split('.')[-1]),
                    len(str(foam_params['cfd_init_time']).split('.')[-1])
                ]))

        if server:
            # Initialize a flow field, otherwise the initial state is empty and learning cannot be performed
            cfd.run_init(foam_root_path, foam_params)
            # Store the initialized flow field result (state) in the _init file to avoid repeated reading and writing
            self.velocity_table_init = utils.read_foam_file(
                foam_root_path + f'/postProcessing/probes/0/U',
                dimension=self.foam_params['num_dimension']
            )
            self.pressure_table_init = utils.read_foam_file(
                foam_root_path + f'/postProcessing/probes/0/p',
                dimension=self.foam_params['num_dimension']
            )

    def states(self):
        """state_space"""
        # using pressure as a training parameter
        if self.state_params['type'] == 'pressure':
            return dict(
                type='float',
                shape=(int(self.state_params['probe_info'].shape[0]), )
            )

        # using velocity as a training parameter
        elif self.state_params['type'] == 'velocity':
            if self.foam_params['num_dimension'] == 2:
                return dict(
                    type='float',
                    shape=(int(2 * self.state_params['probe_info'].shape[0]), )
                )
            elif self.foam_params['num_dimension'] == 3:
                return dict(
                    type='float',
                    shape=(int(3 * self.state_params['probe_info'].shape[0]), )
                )
            else:
                assert 0, 'Simulation type error'
        else:
            assert 0, 'No define state type error'


    def actions(self):
        """action_space"""
        return dict(
            type='float',
            #shape=(len(str(self.agent_params['minmax_value'][0])), ),
            shape=(1, ),
            min_value=self.agent_params['minmax_value'][0],
            max_value=self.agent_params['minmax_value'][1]
        )


    def execute(self, actions=None):
        """Run one timestep of the environment's dynamics."""
        self.trajectory_start_time = time()
        self.num_trajectory += 1
        if actions is None:
            print("carefully, no action given; by default, no jet!")

        # Record the actions of each trajectory, reset the empty list each time the environment is instantiated,
        # and overwrite the update each time execute() is executed
        self.actions_sequence = np.append(self.actions_sequence, actions)

        # Modify the actions of each trajectory: start_actions, end_actions
        if self.num_trajectory < 1.5  :
            self.start_actions=[0]
            self.end_actions = [self.actions_sequence[0]]
        else:
            self.start_actions=[self.actions_sequence[-2]]
            self.end_actions=[self.actions_sequence[-1]]

        start_time_float = np.around(
            float(self.cfd_init_time_str) + (self.num_trajectory - 1) * self.agent_params['interaction_period'],
            decimals=self.decimal
        )
        end_time_float = np.around(start_time_float + self.agent_params['interaction_period'], decimals=self.decimal)

        # Find the current latest time folder, as startTime, to specify the action write folder path
        start_time_filename, start_time_path = utils.get_current_time_path(self.foam_root_path)

        # Change the start_action issued by the agent to the corresponding time folder
        utils.dict2foam(
            start_time_path,
            utils.actions2dict(self.agent_params['entry_dict_q0'], self.agent_params['variables_q0'], self.start_actions)
        )

        # Change the end_action issued by the agent to the corresponding time folder
        utils.dict2foam(
            start_time_path,
            utils.actions2dict(self.agent_params['entry_dict_q1'], self.agent_params['variables_q1'], self.end_actions)
        )

        start_time=[start_time_float]
        # Change the t0 issued by the agent to the corresponding time folder
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

        # Read velocity(state) file
        self.probe_velocity_df = utils.read_foam_file(
            self.foam_root_path + f'/postProcessing/probes/{start_time_filename}/U',
            dimension=self.foam_params['num_dimension']
        )

        # Read pressure file (state)
        self.probe_pressure_df = utils.read_foam_file(
                self.foam_root_path + f'/postProcessing/probes/{start_time_filename}/p',
            dimension=self.foam_params['num_dimension']
        )

        # Read the forces.dat file and output it directly in the form of total force (reward)
        self.force_df = utils.resultant_force(
            utils.read_foam_file(
            self.foam_root_path + f'/postProcessing/forcesIncompressible/{start_time_filename}/forces.dat'
            )
        )

        # Read the force coefficient forceCoeffs.dat file (reward)
        self.force_Coeffs_df = utils.read_foam_file(
            self.foam_root_path + f'/postProcessing/forceCoeffsIncompressible/{start_time_filename}/forceCoeffs.dat'
        )

        # Links all full cycle historical force and force coefficient data prior to the current trajectory
        if self.num_trajectory < 1.5:
            self.history_force_df = self.force_df
            self.history_force_Coeffs_df = self.force_Coeffs_df
        else:
            self.history_force_df = pd.concat([self.history_force_df, self.force_df[1:]]).reset_index(drop=True)
            self.history_force_Coeffs_df = pd.concat(
                [self.history_force_Coeffs_df, self.force_Coeffs_df[1:]]
            ).reset_index(drop=True)


        # Use the last line of the result file as the next state
        if self.state_params['type'] == 'pressure':
            next_state = self.probe_pressure_df.iloc[-1, 1:].to_numpy()
        elif self.state_params['type'] == 'velocity':
            next_state = self.probe_velocity_df.iloc[-1, 1:].to_numpy()
        else:
            next_state = False
            assert next_state, 'No define state type'
        self.state_data = np.append(self.state_data, next_state)

        # Calculate the reward value
        reward = self.reward_function()
        # Record the reward value of each trajectory and episode
        self.trajectory_reward = np.append(self.trajectory_reward, reward)
        self.episode_reward += reward
        print(self.num_trajectory,self.start_actions,self.end_actions,reward)

        # Termination condition
        terminal = False

        self.trajectory_end_time = time()

        # Used to pass the data generated during the step() process
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

        return next_state, terminal, reward

    @abstractmethod
    def reward_function(self):
        """Define reward and formulate it as an abstract method, which means that
        when instantiating the OpenFoam class, the reward_function function must be overridden"""
        pass


    def reset(self):
        """Resets the environment to an initial state and returns the initial observation."""
        if self.num_episode < 0.5:
            os.makedirs(self.foam_root_path + '/record')
        else:
            # Extract the optimal action in the entire episode, skip the first initialization process
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

        # Output all actions up to the current trajectory
        if self.num_episode == 1:
            self.all_episode_actions = pd.DataFrame(self.actions_sequence)
            self.all_episode_trajectory_reward = pd.DataFrame(self.trajectory_reward)
        else:
            self.all_episode_actions[self.num_episode - 1] = pd.DataFrame(self.actions_sequence)
            self.all_episode_trajectory_reward[self.num_episode - 1] = pd.DataFrame(self.trajectory_reward)

        # Save action, reward and lift-drag information
        self.all_episode_actions.to_csv(
            self.foam_root_path + '/record/all_episode_actions.csv', index=False, header=False
        )

        self.all_episode_trajectory_reward.to_csv(
            self.foam_root_path + '/record/all_episode_trajectory_reward.csv', index=False, header=False
        )
        self.history_force_Coeffs_df.to_csv(
            self.foam_root_path + f'/record/history_force_Coeffs_df_{self.num_episode}.csv', index=False, header=False
        )
        self.info_list=pd.DataFrame(self.info_list)
        self.info_list.to_csv(
            self.foam_root_path + f'/record/info_list_{self.num_episode}.csv'
        )
        # update episode
        self.num_episode += 1
        # Reset the trajectory in an episode
        self.num_trajectory = 0
        # Reset cumulative reward
        self.episode_reward = 0
        # Reset the action sequence in the episode
        self.actions_sequence = []
        # Reset the reward sequence in the trajectory
        self.trajectory_reward = []
        # Reset the actual action sequence of actions
        self.decorated_actions_sequence = []
        # Reset historical force and force coefficients
        self.history_force_df = pd.DataFrame()
        self.history_force_Coeffs_df = pd.DataFrame()
        self.info_list = []
        # TODO The flow field needs to be initialized, delete all files, but do not need to
        #  reset the control dictionary, because the set calculation time is already included in the step() function
        # Delete all time folders except 0 and the initial flow field folder and the postprocessing folder postProcessing
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

        # Put the initial state into the state array
        self.state_data = np.append(self.state_data, init_state)

        return init_state


    def force_coeffs_sliding_average(self, sliding_time_interval):
        # Calculate the number of sampling time points
        sampling_num = int(sliding_time_interval / self.foam_params['delta_t'])
        # Calculate the sliding average of the lift coefficient over a vortex shedding cycle        if self.history_force_Coeffs_df.shape[0] <= sampling_num:
        if self.history_force_Coeffs_df.shape[0] <= sampling_num:
            sliding_average_cd = np.mean(signal.savgol_filter(self.history_force_Coeffs_df.iloc[:, 2],49,0))
            sliding_average_cl = np.mean(signal.savgol_filter(self.history_force_Coeffs_df.iloc[:, 3],49,0))
        else:
            sliding_average_cd = np.mean(signal.savgol_filter(self.history_force_Coeffs_df.iloc[-sampling_num:, 2],49,0))
            sliding_average_cl = np.mean(signal.savgol_filter(self.history_force_Coeffs_df.iloc[-sampling_num:, 3],49,0))
        return sliding_average_cd, sliding_average_cl


