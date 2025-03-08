# coding = UTF-8
import os
import re
import subprocess
import argparse
from hanshu.openfoam import utils
# import utils
#import utils

def run(num_trajectory, path, foam_params, agent_interaction_period, purgeWrite_numbers,writeInterval,deltaT,start_time, end_time):
    # 应特别小心controlDict中startFrom是否为startTime，如果是latestTime则会出现控制不了开始时间的状况，但一般来讲也没必要控制开始时间，都是从最新步算起
    control_dict_path = path + '/system/controlDict'
    assert isinstance(end_time, (int, float)), f'TypeError: end_time must be int or float type'
    decompose_par_dict_path = path + '/system/decomposeParDict'
    # 设置并行计算参数
    with open(decompose_par_dict_path, 'r+') as f:
        content = f.read()
        content = re.sub(f'(numberOfSubdomains\s+)\d+;', f'\g<1>{foam_params["num_processor"]};', content)
        f.seek(0)
        f.truncate()
        f.write(content)

    with open(control_dict_path, 'r+') as f:
        content = f.read()
        if start_time == 'latestTime':
            content = re.sub(f'(startFrom\s+).*;', f'\g<1>latestTime;', content)
        elif isinstance(start_time, (int, float)):
            content = re.sub(f'(startFrom\s+).*;', f'\g<1>startTime;', content)
            content = re.sub(f'(startTime\s+).+;', f'\g<1>{start_time};', content)
        else:
            assert False, f'TypeError: start_time must be int, float or specific strings type'
        content = re.sub(f'(endTime\s+).*;', f'\g<1>{end_time};', content)
        content = re.sub(f'(writeInterval\s+).*;', f'\g<1>{agent_interaction_period};', content)
        content = re.sub(f'(purgeWrite\s+).*;', f'\g<1>{purgeWrite_numbers};', content)
        content = re.sub(f'(deltaT\s+).*;', f'\g<1>{deltaT};', content)
        f.seek(0)
        f.truncate()
        f.write(content)

    #终端命令
    # if foam_params['verbose']:
    #     print(11)
    #     subprocess.Popen(
    #          f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + f'pimpleFoam > /dev/null',
    #         shell=True, executable='/bin/bash'
    #     )

    # subprocess.run(
    #     f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + f'pimpleFoam > /dev/null',
    #     shell=True, executable='/bin/bash'
    # )

    if foam_params['verbose']:
        if num_trajectory < 1.5:
            subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + 'decomposePar -force > /dev/null',shell=True, check=True, executable='/bin/bash')
            mpi_process = subprocess.Popen(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',shell=True, executable='/bin/bash')
            mpi_process.communicate()
            subprocess.run(
                f'cd {path}' + ' && ' + foam_params["of_env_init"] + ' && ' + 'reconstructPar > /dev/null',
                shell=True, check=True, executable='/bin/bash'
            )
            # subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + 'cat system/log.pimpleFoam | grep \'of mass\' | cut -d\' \' -f9 | tr -d \'(\'  > postProcessing/disaplacement',shell=True, check=True, executable='/bin/bash')
        else:
            mpi_process = subprocess.Popen(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',shell=True, executable='/bin/bash')
            mpi_process.communicate()
            subprocess.run(
                f'cd {path}' + ' && ' + foam_params["of_env_init"] + ' && ' + 'reconstructPar > /dev/null',
                shell=True, check=True, executable='/bin/bash'
            )
            # subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + 'cat system/log.pimpleFoam | grep \'of mass\' | cut -d\' \' -f9 | tr -d \'(\'  > postProcessing/disaplacement',shell=True, check=True, executable='/bin/bash')
    else:
        if num_trajectory < 1.5:
            subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + 'decomposePar -force > /dev/null',shell=True, check=True, executable='/bin/bash')
            mpi_process = subprocess.Popen(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',shell=True, executable='/bin/bash')
            mpi_process.communicate()
            # subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + 'cat system/log.pimpleFoam | grep \'of mass\' | cut -d\' \' -f9 | tr -d \'(\'  > postProcessing/disaplacement',shell=True, check=True, executable='/bin/bash')
        else:
            mpi_process = subprocess.Popen(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',shell=True, executable='/bin/bash')
            mpi_process.communicate()
            # subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + 'cat system/log.pimpleFoam | grep \'of mass\' | cut -d\' \' -f9 | tr -d \'(\'  > postProcessing/disaplacement',shell=True, check=True, executable='/bin/bash')

@utils.timeit('OpenFOAM_init')
def run_init(path, foam_params):
    # 第一步检查是够初始化，即使流场是从稳定场映射而来，也需要计算一段时间以获得初始状态值
    assert foam_params['cfd_init_time'], f'\n\nInitialization before training is compulsory!\n'

    control_dict_path = path + '/system/controlDict'
    decompose_par_dict_path = path + '/system/decomposeParDict'

    # 设置并行计算参数
    with open(decompose_par_dict_path, 'r+') as f:
        content = f.read()
        content = re.sub(f'(numberOfSubdomains\s+)\d+;', f'\g<1>{foam_params["num_processor"]};', content)
        f.seek(0)
        f.truncate()
        f.write(content)

    # 设置of_init/controlDict文件
    with open(control_dict_path, 'r+') as f:
        content = f.read()
        content = re.sub(f'(application\s+).+;', f'\g<1>{foam_params["solver"]};', content)
        content = re.sub(f'(deltaT\s+).*;', f'\g<1>{foam_params["delta_t"]};', content)
        content = re.sub(f'(startFrom\s+).*;', f'\g<1>startTime;', content)
        content = re.sub(f'(startTime\s+).+;', f'\g<1>0;', content)
        content = re.sub(f'(endTime\s+).+;', f'\g<1>{foam_params["cfd_init_time"]};', content)
        content = re.sub(f'(writeInterval\s+).+;', f'\g<1>{foam_params["cfd_init_time"]};', content)
        content = re.sub(f'(purgeWrite\s+).+;', f'\g<1>0;', content)
        f.seek(0)
        f.truncate()
        f.write(content)

    if foam_params['verbose']:
        subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + 'decomposePar -force > /dev/null',shell=True, check=True, executable='/bin/bash')
        mpi_process = subprocess.Popen(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',shell=True, executable='/bin/bash')
        mpi_process.communicate()
        # subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + 'cat system/log.pimpleFoam | grep \'of mass\' | cut -d\' \' -f9 | tr -d \'(\'  > postProcessing/disaplacement',shell=True, check=True, executable='/bin/bash')
        subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' +'reconstructPar > /dev/null',shell=True, check=True, executable='/bin/bash')
    else:
        subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + 'decomposePar -force > /dev/null',shell=True, check=True, executable='/bin/bash')
        mpi_process = subprocess.Popen(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',shell=True, executable='/bin/bash')
        mpi_process.communicate()
        # subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' + 'cat system/log.pimpleFoam | grep \'of mass\' | cut -d\' \' -f9 | tr -d \'(\'  > postProcessing/disaplacement',shell=True, check=True, executable='/bin/bash')
        subprocess.run(f'cd {path}' + ' && ' + foam_params['of_env_init'] + ' && ' +'reconstructPar > /dev/null',shell=True, check=True, executable='/bin/bash')



