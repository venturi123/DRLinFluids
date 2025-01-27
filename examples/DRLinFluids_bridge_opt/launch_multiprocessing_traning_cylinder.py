# coding = UTF-8
import argparse
import re
from tensorforce import Runner, Agent,Environment
import envobject_cylinder
import os

# define parameters
parser = argparse.ArgumentParser()
#parser.add_argument('-n', '--num-episodes', required=True, type=int, help='Number of episodes')
#parser.add_argument(
#    '-m', '--max-episode-timesteps', required=True, type=int, help='Maximum number of timesteps #per episode'
#)
shell_args = vars(parser.parse_args())
shell_args['num_episodes']=1000
shell_args['max_episode_timesteps']=2

number_servers=3
nstate=1
naction=1
foam_params = {
    'delta_t': 0.0005,
    'solver': 'pimpleFoam',
    'num_processor': 8,
    'of_env_init': '. /opt/openfoam8/etc/bashrc',
    'cfd_init_time': 0.001,
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
    #'deltaA': 0.05,
    'minmax_value': (40, 160),
    'interaction_period': 0.01,
    'purgeWrite_numbers': 0,
    'writeInterval': 0.01,
    'deltaT': 0.0005,
    'variables_q0': ('x',),
    'verbose': False,
    # "zero_net_Qs": True,
}
state_params = {
    'type': 'pressure'
}

# Pre-defined or custom environment
root_path = os.getcwd()
env_name_list = sorted([envs for envs in os.listdir(root_path) if re.search(r'^env\d+$', envs)])
environments = []
for env_name in env_name_list:
    env = envobject_cylinder.FlowAroundCylinder2D(
        foam_root_path='/'.join([root_path, env_name]),
        foam_params=foam_params,
        agent_params=agent_params,
        state_params=state_params,
    )
    environments.append(env)

use_best_model = True
if use_best_model:
    evaluation_environment = environments.pop()
else:
    evaluation_environment = None

network_spec = [
    dict(type='dense', size=512,activation='tanh'),
    dict(type='dense', size=512,activation='tanh')
]
baseline_spec = [
   dict(type='dense', size=512,activation='tanh'),
    dict(type='dense', size=512,activation='tanh')
]

# Instantiate a Tensorforce agent
agent = Agent.create(
    # states=env.states(),
    # actions=env.actions(),
    # max_episode_timesteps=shell_args['max_episode_timesteps'],
    agent='dqn',
    environment=env,max_episode_timesteps=shell_args['max_episode_timesteps'],
    batch_size=20,
     network=network_spec,
    learning_rate=0.005,state_preprocessing=None,
    entropy_regularization=0.1, likelihood_ratio_clipping=0.2, subsampling_fraction=0.2,
    predict_terminal_values=True,
    discount=0.97,
    # baseline=dict(type='1', size=[32, 32]),
    baseline=baseline_spec,
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=5e-3
        ),
        num_steps=1
    ),
    multi_step=1,
    parallel_interactions=number_servers,
    saver=dict(directory=os.path.join(os.getcwd(), 'saved_models/checkpoint'),frequency=1  
    # save checkpoint every 100 updates
    ),
    summarizer=dict(
        directory='summary',
        # list of labels, or 'all'
        labels=['entropy', 'kl-divergence', 'loss', 'reward', 'update-norm']
    ),
)
# print(agent.get_architecture())
# agent = Agent.load(directory='saved_models/checkpoint')

print('Agent defined DONE!')

# Train for 3000 episodes, each episode contain 100 actions
# runner = Runner(agent=agent, environment=env,)
runner = Runner(
    agent=agent,
    environments=environments,
    max_episode_timesteps=shell_args['max_episode_timesteps'],
    evaluation=use_best_model,
    remote='multiprocessing',
)
print('Runner defined DONE!')

# runner.run(episodes=500, max_episode_timesteps=80)
runner.run(num_episodes=shell_args['num_episodes'],
           save_best_agent ='best_model',
           #ssync_episodes=True,
           )
runner.close()

for environment in environments:
    environment.close()

