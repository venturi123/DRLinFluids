import numpy as np
from scipy import signal

from DRLinFluids.environments_tianshou import OpenFoam_tianshou


class FlowAroundSquare2D(OpenFoam_tianshou):
    def reward_function(self):
        vortex_shedding_period = 0.025
        drug_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[0]
        lift_coeffs_sliding_average = self.force_coeffs_sliding_average(vortex_shedding_period)[1]
        print(3.205 - drug_coeffs_sliding_average,0.1 * np.abs(lift_coeffs_sliding_average))
        return   3.205 - drug_coeffs_sliding_average - 0.1 * np.abs(lift_coeffs_sliding_average)


    def agent_actions_decorator(self, actions):
        if self.num_trajectory < 1.5:
            new_action = 0.4 * actions
        else:
            new_action = np.array(self.decorated_actions_sequence[self.num_trajectory - 2]) \
                         + 0.4 * (np.array(actions) - np.array(self.decorated_actions_sequence[self.num_trajectory - 2]))
            # new_action = 0.1 * actions
        return new_action
        # return np.array((1, ))
