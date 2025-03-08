# coding = UTF-8
import os
import re
import shutil
from importlib import import_module

import numpy as np
from tensorforce import Environment

# import cfd
# import utilities

cfd = import_module('cfd')
utilities = import_module('utilities')


# 自定义interactive environment
class Env(Environment):
    def __init__(self, path_root, jet_info, probe_info, learning_params, simu_type=2):
        
        self.path_root = path_root
        # jet的位置信息，jet的速度最大值、最小值
        self.jet_info = jet_info
        # probe的位置信息
        self.probe_info = probe_info
        # 定义interaction的数据类型，例如选择流场中的速度值、压力值、力系数等作为state，干预步长INTERVENTION_T，迭代步长DELTA_T
        self.learning_params = learning_params
        # 设定simulation类型（2D OR 3D）
        self.simu_type = simu_type

        # 只有当设置的数值在适合的范围内时，才会采用，若越界,直接采用所有数据进行计算,若小于等于0，则设为1
        # 需要注意的是，self.learning_params['number_history_value']一定要为int类型，否则后续切片操作会报错
        if self.learning_params['number_history_value'] > 0:
            self.learning_params['number_history_value'] = np.min([self.learning_params['number_history_value'], (self.learning_params['INTERVENTION_T'] / self.learning_params['DELTA_T']) + 1])
            self.learning_params['number_history_value'] = int(self.learning_params['number_history_value'])
        else:
            self.learning_params['number_history_value'] = int(1)

        # 初始化一个流场，不然初始state为空，无法进行学习
        cfd.run_init(path_root, learning_params, turbo_sleep_time=-1)

        # 初始化结果表（DataFrame），将稳定后的流场作为初始值
        # 读取输入的初始化流场时刻，调整其格式
        self.init_satrt_time = str(float(learning_params['INIT_START_TIME'])).rstrip('0').rstrip('.')
        # 读取velocity(state)文件
        self.velocity_table = utilities.read_foam_file(self.path_root + f'/postProcessing/probes/0/U')
        # 读取pressure(state)文件
        self.pressure_table = utilities.read_foam_file(self.path_root + f'/postProcessing/probes/0/p')
        # 读取forces.dat(reward)文件，直接输出为总分力形式
        self.force_table = utilities.resultant_force(utilities.read_foam_file(self.path_root + '/postProcessing/forcesIncompressible/0/forces.dat'))
        
        # 将初始化流场结果储存到_init文件中，避免重复读写
        self.velocity_table_init = self.velocity_table
        self.pressure_table_init = self.pressure_table
        self.force_table_init = self.force_table
        
        # 保存INTERVENTION_T和INIT_START_TIME的小数位长度最大值，后续在读取文件时，会根据该变量，自动截取有效位数，避免在进制转换时出现误差而导致不能正确读取文件
        self.decimal = int(np.max([len(str(learning_params['INTERVENTION_T']).split('.')[-1]), len(str(learning_params['INIT_START_TIME']).split('.')[-1])]))

        print(f'path_root: {path_root}')
        print(f'jet_info: {jet_info}')
        print(f'learning_params: {learning_params}')
        print(f'simu_type: {simu_type}')
        print(f'init_satrt_time: {self.init_satrt_time}')
        print(f'decimal: {self.decimal}')

    # 定义state space
    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are available simultaneously.

        Returns: dict of state properties (shape and type).
        """

        # 将pressure作为训练参数
        if self.learning_params['state_type'] == 'pressure':
            return dict(type='float',
                        shape=(int(self.probe_info['position'].shape[0] * self.learning_params['number_history_value']), )
                        )

        # velocity作为训练参数
        elif self.learning_params['state_type'] == 'velocity':
            if self.simu_type == 2:
                return dict(type='float',
                            shape=(int(2 * self.probe_info['position'].shape[0] * self.learning_params['number_history_value']), )
                            )
            elif self.simu_type == 3:
                return dict(type='float',
                            shape=(int(3 * self.probe_info['position'].shape[0] * self.learning_params['number_history_value']), )
                            )
            else:
                assert 0, 'Simulation type error'

        else:
            assert 0, 'No define state type error'


    # 定义action space
    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are available simultaneously.

        Returns: dict of action properties (continuous, number of actions).
        """

        # NOTE: we could also have several levels of dict in dict, for example:
        # return { str(i): dict(continuous=True, min_value=0, max_value=1) for i in range(self.n + 1) }

        return dict(type='float',
                    shape=(len(self.jet_info['name']), ),
                    min_value=self.jet_info['min_value'],
                    max_value=self.jet_info['max_value'])


    def execute(self, actions=None):

        # print("--- call execute ---")
        if actions is None:
            print("carefull, no action given; by default, no jet!")
        # 打印当前步action
        # print(f'Number of trajectory: {self.trajectory}')

        # 必须提前处理，而不能留到后面读取文件的时候处理，因为中间还有个cfd.run()，会涉及实际计算时间
        start_time = np.around(float(self.init_satrt_time) + self.trajectory * self.learning_params['INTERVENTION_T'], decimals=self.decimal)
        end_time = np.around(start_time + self.learning_params['INTERVENTION_T'], decimals=self.decimal)

        # print(f'start_time: {start_time}')
        # print(f'end_time: {end_time}')

        # 执行一次trajectory
        cfd.run(self.path_root, self.learning_params, self.jet_info, actions, start_time, end_time, turbo_sleep_time=-1)
        self.trajectory += 1

        # 注意str(float(start_time)).rstrip('0').rstrip('.')的写法，用于去除可能因为浮点数运算而产生的多余的零和小数点
        # 先将start_time强制转换为浮点数，是因为如果出现末尾为0的整数，例如0、10等，将会出错
        time_file_name = str(float(start_time)).rstrip('0').rstrip('.')
        # 之所以不能直接使用上述语句，是因为浮点数运算不仅会产生多余的0和小数点，而且会因为二进制与十进制的转换，产生计算误差
        # 好在此处的start_time是通过初始化时间加

        # 读取velocity(state)文件
        self.velocity_table = utilities.read_foam_file(self.path_root + f'/postProcessing/probes/{time_file_name}/U')

        # 检查设定的选取历史数据值是否存在或越界，如果存在以上两种情况，则计算所有数据
        # if (self.learning_params.has_key('number_history_value') == False) or (self.learning_params['number_history_value'] > self.velocity_table.shape[0]):
        #     # self.learning_params['number_history_value'] = ((end_time - start_time) / self.learning_params['INTERVENTION_T']) + 1
        #     self.learning_params['number_history_value'] = self.velocity_table.shape[0]

        # 读取pressure(state)文件
        self.pressure_table = utilities.read_foam_file(self.path_root + f'/postProcessing/probes/{time_file_name}/p')
        # 读取forces.dat(reward)文件，直接输出为总分力形式
        self.force_table = utilities.resultant_force(utilities.read_foam_file(self.path_root + f'/postProcessing/forcesIncompressible/{time_file_name}/forces.dat'))

        # 将结果文件的最后一行作为下一个state状态
        if self.learning_params['state_type'] == 'pressure':
            next_state = self.pressure_table.iloc[-1,1:]
        elif self.learning_params['state_type'] == 'velocity':
            next_state = self.velocity_table.iloc[-1,1:]
        else:
            next_state = -1
            assert 0, 'No define state type'

        reward = self.compute_reward()

        self.total_reward += reward

        # TODO 中止条件需要认真考虑，可以考虑到total_reward低于某个值的时候停止
        terminal = False

        return(next_state, terminal, reward)


    # 定义reward
    def compute_reward(self):
        # NOTE: reward should be computed over the whole number of iterations in each execute loop

        # 切片最后的number_steps_execution步计算结果放入到values_drag_in_last_execute中，在对其去均值+0.159
        if self.learning_params['reward_function_type'] == 'plain_drag':  # a bit dangerous, may be injecting some momentum
            values_drag_in_last_execute = self.force_table[:, 1]
            # TODO: the 0.159 value is a proxy value corresponding to the mean drag when no control
            return (np.mean(values_drag_in_last_execute.iloc[:,1]) + 15)

        # TODO 需要在OF中编写后处理utility，初步想法为将旋度场截取一定阈值以上，记录网格编号，再乘以对应体积即可 -> 存在问题，这样计算整个流场的涡旋体积都将被记录
        # elif self.learning_params['reward_function_type'] == 'recirculation_area':
        #     # 计算recirculation区域面积,有一套自己的计算方法,详见probe.py中的RecirculationAreaProbe类
        #     return - self.area_probe.sample(self.u_, self.p_)
        # elif self.learning_params['reward_function_type'] == 'max_recirculation_area':
        #     return self.area_probe.sample(self.u_, self.p_)

        # 仅仅取最后一次的拖拽力作为奖励函数
        elif self.learning_params['reward_function_type'] == 'last_drag':  # a bit dangerous, may be injecting some momentum
            return self.force_table.iloc[1,-1] + 15

        # 综合了升力和拖拽力
        elif self.learning_params['reward_function_type'] == 'drag_plain_lift':  # a bit dangerous, may be injecting some momentum
            # TODO 需要考虑最小值到底取多少
            avg_drag = np.mean(self.force_table.iloc[:,1])
            # avg_lift = np.mean(self.force_df.iloc[:,2])
            return (15 - avg_drag) - (9 - 0.1 * np.var(self.force_table.iloc[:,2])) + 7

        # TODO 待定
        elif self.learning_params['reward_function_type'] == 'max_plain_drag':  # a bit dangerous, may be injecting some momentum
            # values_drag_in_last_execute = self.data_recorder["drag"].get()[-self.number_steps_execution:]
            # return - (np.mean(values_drag_in_last_execute) + 0.159)
            return 1 / np.max(self.force_table.iloc[:,1])

        elif self.learning_params['reward_function_type'] == 'drag_avg_abs_lift':  # a bit dangerous, may be injecting some momentum
            avg_length = min(500, self.learning_params['number_history_value'])
            avg_abs_lift = np.mean(np.absolute(self.force_table.iloc[-avg_length:,2]))
            avg_drag = np.mean(self.force_table.iloc[-avg_length:,1])
            return avg_drag + 0.159 - 0.2 * avg_abs_lift

        # TODO: implement some reward functions that take into account how much energy / momentum we inject into the flow
        else:
            raise RuntimeError("reward function {} not yet implemented".format(self.learning_params['reward_function_type']))

    # 此处的reset是用于episode之间的重置，不是每个trajectory的重置
    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        # print('触发reset')
        # 将初始化备份的流场信息DataFrame读取到新的一轮episode初始值中
        # self.velocity_table = self.velocity_table_init
        # self.probe_pressure_df = self.pressure_table_init
        # self.force_df = self.force_table_init

        # 一个episode中的trajectory
        self.trajectory = 0
        # 累计reward
        self.total_reward = 0

        # TODO 流场需要初始化，删除所有文件，但不用重置控制字典，因为在excute()函数中已经包含了设定计算时间 
        # 删除除0和初始流场文件夹以外所有时间文件夹以及后处理文件夹postProcessing
        # 返回当前工作目录上级目录的所有文件名称：os.listdir('/'.join(self.foam_root_path.split('/')[:-1]))
        for f_name in os.listdir(self.path_root):
            if re.search('^\d+\.?\d*', f_name):
                if (f_name != '0') and (f_name != self.init_satrt_time):
                    shutil.rmtree('/'.join([self.path_root, f_name]))
            elif f_name == 'postProcessing':
                shutil.rmtree('/'.join([self.path_root, f_name]))
            else:
                pass

        if self.learning_params['state_type'] == 'pressure':
            init_state = self.pressure_table_init.iloc[-1,1:]
        elif self.learning_params['state_type'] == 'velocity':
            init_state = self.velocity_table_init.iloc[-1,1:]
        else:
            init_state = -1
            assert 0, 'No define state type'
        return init_state


    def max_episode_timesteps(self):
        return 10

    def close(self):
        super().close()
