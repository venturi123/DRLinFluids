# coding = UTF-8
import os
import re
import subprocess
import time
from importlib import import_module

utilities = import_module('utilities')


def run(path, learning_params, jet_info, boundary_value, start_time, end_time, turbo_sleep_time=-1, verbose=False):

    control_dict_path = path + '/system/controlDict'

    with open(control_dict_path, 'r') as f:
        control_dict_text_org = f.read()

    # 设置startTime
    control_dict_text_1 = re.sub(r'startTime.+;', f'startTime\t{start_time};', control_dict_text_org)

    # 设置endTime
    control_dict_text_2 = re.sub(r'endTime.+;', f'endTime\t\t{end_time};', control_dict_text_1)

    with open(control_dict_path, 'w') as f:
        f.write(control_dict_text_2)

    # 在decomposePar之前先设定好风速条件
    with open(f'{path}/{int(start_time)}/U', 'r') as f:
        u = f.read()

    boundary_No = 0  # 记录边界循环次数，用于对应wind_speed中的值
    # 将边界列表中所有值都更新一遍
    for boundary in jet_info['name']:
        boundary_No += 1
        with open(f'{path}/{int(start_time)}/U', 'w+') as f:
            f.write(re.sub(f'{boundary}\n(.*\n)+?\s*}}', f'{boundary}\n\t{{\n\t\ttype\t\t\tfixedValue;\n\t\tvalue\t\t\tuniform (0 {boundary_value[boundary_No-1]} 0);\n\t}}', u)) 
            f.seek(0)
            u = f.read()

    if verbose:
        subprocess.run(f'cd {path}' + ' && ' + learning_params['OF_INIT'] + ' && ' + 'decomposePar -force', shell=True, check=True, executable='/bin/bash')
        mpi_process = subprocess.Popen(f'cd {path}' + ' && ' + learning_params['OF_INIT'] + ' && ' + f'mpirun -np {learning_params["N_PROCESSOR"]} {learning_params["SOLVER"]} -parallel', shell=True, executable='/bin/bash')
        # subprocess.run(OF_INIT + ' && ' + f'mpirun -np {N_PROCESSOR} {SOLVER} -parallel && ', shell=True, check=True, executable='/bin/bash')
        # 此处必须设置sleep且大于3s，5s更保险，因为在执行mpirun时，采取了非阻塞形式，此时如果不加停顿立即调整进程优先级，则可能会出现读取不到PID导致程序出错中止
        # 若turbo_sleep_time小于等于0，则关闭turbo功能
        if turbo_sleep_time > 0:
            time.sleep(turbo_sleep_time)
            # 调整mpirun优先级，注意运行用户需要在sudoers文件中添加NOPASSWD属性
            subprocess.run(f'ps aux | grep {learning_params["SOLVER"]} | grep -v grep | cut -c 10-14 | xargs sudo renice {learning_params["NICE"]} -p', shell=True, check=True, executable='/bin/bash')
        mpi_process.communicate()
        subprocess.run(f'cd {path}' + ' && ' + learning_params["OF_INIT"] + ' && ' + 'reconstructPar', shell=True, check=True, executable='/bin/bash')
    else:
        subprocess.run(f'cd {path}' + ' && ' + learning_params['OF_INIT'] + ' && ' + 'decomposePar -force > decompose.log', shell=True, check=True, executable='/bin/bash')
        mpi_process = subprocess.Popen(f'cd {path}' + ' && ' + learning_params['OF_INIT'] + ' && ' + f'mpirun -np {learning_params["N_PROCESSOR"]} {learning_params["SOLVER"]} -parallel > case.log', shell=True, executable='/bin/bash')
        # subprocess.run(OF_INIT + ' && ' + f'mpirun -np {N_PROCESSOR} {SOLVER} -parallel && ', shell=True, check=True, executable='/bin/bash')
        # 此处必须设置sleep且大于3s，5s更保险，因为在执行mpirun时，采取了非阻塞形式，此时如果不加停顿立即调整进程优先级，则可能会出现读取不到PID导致程序出错中止
        # 若turbo_sleep_time小于等于0，则关闭turbo功能
        if turbo_sleep_time > 0:
            time.sleep(turbo_sleep_time)
            # 调整mpirun优先级，注意运行用户需要在sudoers文件中添加NOPASSWD属性
            subprocess.run(f'ps aux | grep {learning_params["SOLVER"]} | grep -v grep | cut -c 10-14 | xargs sudo renice {learning_params["NICE"]} -p', shell=True, check=True, executable='/bin/bash')
        mpi_process.communicate()
        subprocess.run(f'cd {path}' + ' && ' + learning_params["OF_INIT"] + ' && ' + 'reconstructPar > reconstruct.log', shell=True, check=True, executable='/bin/bash')


@utilities.timeit('OpenFOAM_init')
def run_init(path, learning_params, turbo_sleep_time=-1, verbose=False):

    # path = os.getcwd()
    control_dict_path = path + '/system/controlDict'
    decompose_par_dict = path + '/system/decomposeParDict'
    run_time = learning_params['INIT_START_TIME']

    # 设置并行计算参数
    with open(f'{os.getcwd()}/TensorFOAM/of_init_files/decomposeParDict', 'r') as f:
        decompose_par_dict_org = f.read()
    
    with open(decompose_par_dict, 'w') as f:
        f.write(re.sub(r'numberOfSubdomains\s+\d+', f'numberOfSubdomains\t{learning_params["N_PROCESSOR"]}', decompose_par_dict_org))

    # 读取of_init/controlDict文件
    with open(f'{os.getcwd()}/TensorFOAM/of_init_files/controlDict', 'r') as f:
        control_dict_text_org = f.read()

    # 设置求解器
    control_dict_text_1 = re.sub(r'application\s+.+;', f'application\t{learning_params["SOLVER"]};', control_dict_text_org)
    # 设置迭代时间步deltaT，用户输入
    control_dict_text_2 = re.sub(r'\ndeltaT\s+\d+\.?\d*;', f'\ndeltaT\t\t{learning_params["DELTA_T"]};', control_dict_text_1)
    # 设置writeInterval为干扰周期INTERVENTION_T，因为只需要写最后一步的数据即可
    control_dict_text_3 = re.sub(r'writeInterval\s+\d+', f'writeInterval\t{learning_params["INTERVENTION_T"]}', control_dict_text_2)
    # 设置purgeWrite = 1，确保只保留最后一步的结果，以节省磁盘空间并reconstruct最后一步，因为在训练过程中不需要流场无关细节，相关细节已经保存在postProcess中
    control_dict_text = re.sub(r'purgeWrite\s+\d+', 'purgeWrite\t1', control_dict_text_3)

    control_dict_text_4 = re.sub(r'startTime.+;', 'startTime\t0;', control_dict_text)
    control_dict_text = re.sub(r'endTime.+;', f'endTime\t\t{run_time};', control_dict_text_4)

    with open(control_dict_path, 'w') as f:
        f.write(control_dict_text)

    if verbose:
        subprocess.run(f'cd {path}' + ' && ' + learning_params['OF_INIT'] + ' && ' + 'decomposePar -force', shell=True, check=True, executable='/bin/bash')
        mpi_process = subprocess.Popen(f'cd {path}' + ' && ' + learning_params['OF_INIT'] + ' && ' + f'mpirun -np {learning_params["N_PROCESSOR"]} {learning_params["SOLVER"]} -parallel', shell=True, executable='/bin/bash')
        # subprocess.run(OF_INIT + ' && ' + f'mpirun -np {N_PROCESSOR} {SOLVER} -parallel && ', shell=True, check=True, executable='/bin/bash')
        # 此处必须设置sleep且大于3s，5s更保险，因为在执行mpirun时，采取了非阻塞形式，此时如果不加停顿立即调整进程优先级，则可能会出现读取不到PID导致程序出错中止
        # 若turbo_sleep_time小于等于0，则关闭turbo功能
        if turbo_sleep_time > 0:
            time.sleep(turbo_sleep_time)
            # 调整mpirun优先级，注意运行用户需要在sudoers文件中添加NOPASSWD属性
            subprocess.run(f'ps aux | grep {learning_params["SOLVER"]} | grep -v grep | cut -c 10-14 | xargs sudo renice {learning_params["NICE"]} -p', shell=True, check=True, executable='/bin/bash')
        mpi_process.communicate()
        subprocess.run(f'cd {path}' + ' && ' + learning_params["OF_INIT"] + ' && ' + 'reconstructPar', shell=True, check=True, executable='/bin/bash')
    else:
        subprocess.run(f'cd {path}' + ' && ' + learning_params['OF_INIT'] + ' && ' + 'decomposePar -force > decompose.log', shell=True, check=True, executable='/bin/bash')
        mpi_process = subprocess.Popen(f'cd {path}' + ' && ' + learning_params['OF_INIT'] + ' && ' + f'mpirun -np {learning_params["N_PROCESSOR"]} {learning_params["SOLVER"]} -parallel > case.log', shell=True, executable='/bin/bash')
        # subprocess.run(OF_INIT + ' && ' + f'mpirun -np {N_PROCESSOR} {SOLVER} -parallel && ', shell=True, check=True, executable='/bin/bash')
        # 此处必须设置sleep且大于3s，5s更保险，因为在执行mpirun时，采取了非阻塞形式，此时如果不加停顿立即调整进程优先级，则可能会出现读取不到PID导致程序出错中止
        # 若turbo_sleep_time小于等于0，则关闭turbo功能
        if turbo_sleep_time > 0:
            time.sleep(turbo_sleep_time)
            # 调整mpirun优先级，注意运行用户需要在sudoers文件中添加NOPASSWD属性
            subprocess.run(f'ps aux | grep {learning_params["SOLVER"]} | grep -v grep | cut -c 10-14 | xargs sudo renice {learning_params["NICE"]} -p', shell=True, check=True, executable='/bin/bash')
        mpi_process.communicate()
        subprocess.run(f'cd {path}' + ' && ' + learning_params["OF_INIT"] + ' && ' + 'reconstructPar > reconstruct.log', shell=True, check=True, executable='/bin/bash')


if __name__ == "__main__":
    learning_params = { 'state_type': 'pressure',
        'reward_function_type': 'drag_plain_lift',
        # 选取读取state历史数，一般只取最后一次观察到的数据
        'number_history_value': 1,
        'INTERVENTION_T': 1,
        'DELTA_T': 0.005,
        'SOLVER': 'pisoFoam',
        'N_PROCESSOR': 16,
        'OF_INIT': 'source /opt/rh/python27/enable && source /opt/rh/devtoolset-7/enable && source $HOME/OpenFOAM/OpenFOAM-8/etc/bashrc WM_LABEL_SIZE=64 WM_MPLIB=OPENMPI FOAMY_HEX_MESH=yes',
        'NICE': -10
    }

    run_init(os.getcwd(), learning_params, run_time=5)
