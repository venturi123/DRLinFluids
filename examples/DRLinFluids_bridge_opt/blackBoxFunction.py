from DRLinFluids import utils
from DRLinFluids.utils import dict2foam,actions2dict
from DRLinFluids.cfd import run
import numpy as np
import os
from scipy import signal
import re
import shutil



#将整个cfd计算映射到目标函数的过程都写在黑盒函数black_box_function中
#这里面黑箱函数就是CFD的过程  即输入一个action得到一个后处理后的奖励函数Rewardfuction,我们再优化问题中将这个Rewardfuction成为目标函数ObjectiveFunction


def black_box_function(omega):
    """Function with unknown internals we wish to maximize. 解耦

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    #这里黑箱的过程是必须要有的，但是具体的表达形式我们不知道，其实这个黑箱就是cfd的过程，每次迭代进行贝叶斯优化的时候就是在走一边黑箱（CFD过程）
    def reward_function():
        vortex_shedding_period = agent_params['interaction_period']

        meandrug_coeffs_sliding_average = meanforce_coeffs_sliding_average(vortex_shedding_period)[0]
        # lift_coeffs_sliding_average= signal.savgol_filter(force_coeffs_sliding_average(vortex_shedding_period)[1],0,3)
        meanlift_coeffs_sliding_average = meanforce_coeffs_sliding_average(vortex_shedding_period)[1]
        meanmoment_coeffs_sliding_average = meanforce_coeffs_sliding_average(vortex_shedding_period)[2]

        stddrug_coeffs_sliding_average = stdforce_coeffs_sliding_average(vortex_shedding_period)[0]
        stdlift_coeffs_sliding_average = stdforce_coeffs_sliding_average(vortex_shedding_period)[1]
        stdmoment_coeffs_sliding_average = stdforce_coeffs_sliding_average(vortex_shedding_period)[2]
        print(-1000 * stdlift_coeffs_sliding_average)
        return -1000 * stdlift_coeffs_sliding_average
    def meanforce_coeffs_sliding_average(sliding_time_interval):
        # 计算采样时间点数
        sampling_num = int(sliding_time_interval / foam_params['delta_t'])
        # 计算一个旋涡脱落周期内的升力系数滑动平均值
        if history_force_Coeffs_df.shape[0] <= sampling_num:
            sliding_average_cd = np.mean(signal.savgol_filter(history_force_Coeffs_df.iloc[806:, 2], 1,
                                                              0))  # 0.4s 按照delataT是0.0005计算这806代表0.4s之后的数据
            sliding_average_cl = np.mean(signal.savgol_filter(history_force_Coeffs_df.iloc[806:, 3], 1, 0))
            sliding_average_cm = np.mean(signal.savgol_filter(history_force_Coeffs_df.iloc[806:, 1], 1, 0))

        else:
            sliding_average_cd = np.mean(
                signal.savgol_filter(history_force_Coeffs_df.iloc[1006:, 2], 1, 0))
            sliding_average_cl = np.mean(
                signal.savgol_filter(history_force_Coeffs_df.iloc[1006:, 3], 1, 0))
            sliding_average_cm = np.mean(
                signal.savgol_filter(history_force_Coeffs_df.iloc[1006:, 1], 1, 0))

        return sliding_average_cd, sliding_average_cl, sliding_average_cm


    def stdforce_coeffs_sliding_average(sliding_time_interval):
        # 计算采样时间点数
        sampling_num = int(sliding_time_interval / foam_params['delta_t'])
        # 计算一个旋涡脱落周期内的升力系数滑动平均值
        if history_force_Coeffs_df.shape[0] <= sampling_num:
            sliding_average_cd = np.std(signal.savgol_filter(history_force_Coeffs_df.iloc[1006:, 2], 1, 0))
            sliding_average_cl = np.std(signal.savgol_filter(history_force_Coeffs_df.iloc[1006:, 3], 1, 0))
            sliding_average_cm = np.std(signal.savgol_filter(history_force_Coeffs_df.iloc[1006:, 1], 1, 0))

        else:
            sliding_average_cd = np.std(
                signal.savgol_filter(history_force_Coeffs_df.iloc[1006:, 2], 1, 0))

            sliding_average_cl = np.std(
                signal.savgol_filter(history_force_Coeffs_df.iloc[1006:, 3], 1, 0))
            sliding_average_cm = np.std(
                signal.savgol_filter(history_force_Coeffs_df.iloc[1006:, 1], 1, 0))

        return sliding_average_cd, sliding_average_cl, sliding_average_cm
    foam_params = {
        'delta_t': 0.0005,
        'solver': 'pimpleFoam',
        'num_processor': 16,
        'of_env_init': '. /opt/openfoam8/etc/bashrc',
        'num_dimension': 3,
        'verbose': False
    }
    entry_dict_q0 = {
        'U': {
            'JIET_TRAILING': {
                'omega': '{x}',
            }
        }
    }
    agent_params = {
        'entry_dict_q0': entry_dict_q0,
        'minmax_value': (40, 160),
        'interaction_period': 1.0,
        'purgeWrite_numbers': 0,
        'writeInterval': 0.1,
        'deltaT': 0.0005,
        'variables_q0': ('x',),
        'verbose': False,
        # "zero_net_Qs": True,
    }

    foam_root_path = r'/home/dxl/OptInFluidsTest2/test3'



    #actions2dict(agent_params['entry_dict_q0'], agent_params['variables_q0'], omega)
    #question1: don't know type of omega,actions2dict_Function only support iterative variable such as list
    omegalist=[omega]
    dict2foam(
        r'/home/dxl/OptInFluidsTest2/test3/0',
        actions2dict(agent_params['entry_dict_q0'], agent_params['variables_q0'], omegalist)
    )
    #print("finish omega writen")

    #CFD run progress
    #print("startCFD")
    start_time_float = 0.0
    end_time_float = 1.0

    run(
        foam_root_path,
        foam_params,
        agent_params['writeInterval'],
        agent_params['deltaT'],
        start_time_float, end_time_float
    )

    # 这个时候cfd完成了一个迭代过程的运行，就可以进入后处理了还是用reward_function()函数
    force_Coeffs_df = utils.read_foam_file(
        foam_root_path + f'/postProcessing/forceCoeffsIncompressible/0/forceCoeffs.dat'
    )
    #print(force_Coeffs_df)
    history_force_Coeffs_df = force_Coeffs_df
    objectFunction =reward_function()
    # reset direction 重置的代码
    for f_name in os.listdir(foam_root_path):
        if re.search(r'^\d+\.?\d*', f_name):
            if (f_name != '0'):
                shutil.rmtree('/'.join([foam_root_path, f_name]))
        elif f_name == 'postProcessing':
            shutil.rmtree('/'.join([foam_root_path, f_name]))
        else:
            pass
    return objectFunction



"""
actions2dict   check
dict2foam    check
cfd.run     check 

rewardfunction  check

reset code  check

"""
