import numpy as np
from scipy import signal

from hanshu.openfoam.environments import OpenFoam


class FlowAroundSquareCylinder2D(OpenFoam):
    def reward_function(self):
        # # # 设置旋涡脱落周期为8s
        vortex_shedding_period = 0.025
        #         # # # 调用父类函数
        drug_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[0]
        # lift_coeffs_sliding_average= signal.savgol_filter(self.force_coeffs_sliding_average(vortex_shedding_period)[1],0,3)
        lift_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[1]
        print(3.205 - drug_coeffs_sliding_average,0.1 * np.abs(lift_coeffs_sliding_average))
        return   3.205 - drug_coeffs_sliding_average - 0.1 * np.abs(lift_coeffs_sliding_average)


    def agent_actions_decorator(self, actions):
        # 在这里要给actions套一层函数，抽象成函数传递进来，无需考虑new_action的类型，会在parse函数中统一转换
        if self.num_trajectory < 1.5:
            new_action = 0.4 * actions
        else:
            # 注意此处的index之所以是self.trajectory - 2是因为self.trajectory本身就是真实回合数，本回合引索应-1.上一回合引索-2
            new_action = np.array(self.decorated_actions_sequence[self.num_trajectory - 2]) \
                         + 0.4 * (np.array(actions) - np.array(self.decorated_actions_sequence[self.num_trajectory - 2]))
            # new_action = 0.1 * actions
        return new_action
        # return np.array((1, ))
