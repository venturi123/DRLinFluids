import numpy as np

from DRLinFluids_bridge_opt.DRLinFluids import OpenFoam_tensorforce


class FlowAroundCylinder2D(OpenFoam_tensorforce):
    def reward_function(self):
        vortex_shedding_period = self.agent_params['interaction_period']

        meandrug_coeffs_sliding_average = self.meanforce_coeffs_sliding_average(vortex_shedding_period)[0]
        # lift_coeffs_sliding_average= signal.savgol_filter(self.force_coeffs_sliding_average(vortex_shedding_period)[1],0,3)
        meanlift_coeffs_sliding_average = self.meanforce_coeffs_sliding_average(vortex_shedding_period)[1]
        meanmoment_coeffs_sliding_average = self.meanforce_coeffs_sliding_average(vortex_shedding_period)[2]

        stddrug_coeffs_sliding_average = self.stdforce_coeffs_sliding_average(vortex_shedding_period)[0]
        stdlift_coeffs_sliding_average = self.stdforce_coeffs_sliding_average(vortex_shedding_period)[1]
        stdmoment_coeffs_sliding_average = self.stdforce_coeffs_sliding_average(vortex_shedding_period)[2]
        print(-1000 * stdlift_coeffs_sliding_average)
        return -1000 * stdlift_coeffs_sliding_average


    def agent_actions_decorator(self, actions):
        if self.num_trajectory < 1.5:
            new_action = 0.4 * actions
        else:
            new_action = np.array(self.decorated_actions_sequence[self.num_trajectory - 2]) \
                         + 0.4 * (np.array(actions) - np.array(self.decorated_actions_sequence[self.num_trajectory - 2]))
            # new_action = 0.1 * actions
        return new_action
        # return np.array((1, ))
