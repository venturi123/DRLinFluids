# coding = UTF-8
import json
import os
import re
import time
from io import StringIO
import numpy as np
import pandas as pd




# class RingBuffer():
#     """A 1D ring buffer using numpy arrays"""
#     def __init__(self, length):
#         self.data = np.zeros(length, dtype='f')
#         self.index = 0
#
#     def extend(self, x):
#         """adds ndarray x to ring buffer"""
#         x_index = (self.index + np.arange(x.size)) % self.data.size
#         self.data[x_index] = x
#         self.index = x_index[-1] + 1
#
#     def get(self):
#         """Returns the first-in-first-out data in the ring buffer"""
#         idx = (self.index + np.arange(self.data.size)) % self.data.size
#         return self.data[idx]


"""

def freq_domain_analysis(data, const, threshold=0.5, min_distance=30) -> dict:
    if 'interval' in const:
        interval = const['interval']
    else:
        interval = [None, None]
    data = np.array(data)
    num_selected_points = data[interval[0]:interval[1]].shape[0]
    cl_mean = np.mean(data[interval[0]:interval[1]])
    cl_rms = np.sqrt(np.sum(data[interval[0]:interval[1]] ** 2) / num_selected_points)
    cd_mean = np.mean(data[interval[0]:interval[1], 2])

    Cl = data[interval[0]:interval[1]]
    t_s = data[1, 0] - data[0, 0]
    f_s = 1 / t_s
    F = np.fft.fft(Cl)
    f = np.fft.fftfreq(num_selected_points, t_s)
    mask = np.where(f >= 0)
    peaks_index = peakutils.indexes(np.abs(F[mask])/num_selected_points, thres=threshold, min_dist=min_distance)
    peaks_x = np.array(f[peaks_index])
    peaks_y = np.array(np.abs(F[peaks_index])/num_selected_points)
    shedding_frequency = np.sum(peaks_x * peaks_y) / peaks_y.sum()
    strouhal_number = shedding_frequency * const['D'] / const['U']

    result = {
        'Cl_mean': cl_mean,
        'Cl_RMS': cl_rms,
        'Cd_mean': cd_mean,
        'num_selected_points': num_selected_points,
        'num_all_points': data.shape[0],
        'sampling_frequency': f_s,
        'sampling_period': t_s,
        'shedding_frequency': shedding_frequency,
        'strouhal_number': strouhal_number
    }

    return result
"""




"""
def check_ports(host, port, num_ports=0, verbose=True):
    if isinstance(port, int) and not num_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                if verbose:
                    print(f'host {host} on port {port} is AVAILABLE')
                return True
            except:
                if verbose:
                    print(f'host {host} on port {port} is BUSY')
                return False
    elif isinstance(port, list) and not num_ports:
        for crrt_port in port:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((host, crrt_port))
                    if verbose:
                        print(f'host {host} on port {crrt_port} is AVAILABLE')
                except:
                    if verbose:
                        print(f'host {host} on port {crrt_port} is BUSY')
                    return(False)
        if verbose:
            print("all ports available")
        return True
    elif isinstance(port, int) and num_ports:
        list_ports = [ind_server + port for ind_server in range(num_ports)]
        for crrt_port in list_ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((host, crrt_port))
                    if verbose:
                        print(f'host {host} on port {crrt_port} is AVAILABLE')
                except:
                    if verbose:
                        print(f'host {host} on port {crrt_port} is BUSY')
                    return(False)
        if verbose:
            print("all ports available")
        return(True)
    else:
        assert 0, 'check input arguments!'
"""

"""
def best_training_episode(root_path, saver='/best_training_episode'):
    max_episode_reward = []
    env_name_list = sorted([dir for dir in os.listdir(root_path) if re.search('^env\d+$', dir)])
    env_best_path = ['/'.join([root_path, dir, 'best_episode']) for dir in env_name_list]
    os.makedirs(root_path + saver)

    for path in env_best_path:
        max_episode_reward.append(np.max(pd.read_csv(path + '/total_reward.csv', header=None).to_numpy()))

    max_index = max_episode_reward.index(max(max_episode_reward))

    pd.read_csv(env_best_path[max_index] + '/best_actions.csv', header=None).to_csv(root_path + saver + '/all_best_action.csv')
    pd.Series(max_episode_reward).to_csv(root_path + saver + '/max_episode_reward.csv')

    with open(root_path + saver + '/info.txt', 'w') as f:
        f.write(f'best environment of reward is {max_index}')
"""

"""
def yield_cirprobe(diameter, num, decimal=6, centroid=(0, 0), saver=False, indent=4) -> dict:
    a = [i - 0.5 * diameter for i in centroid]
    b = [centroid[0] - 0.5 * diameter, centroid[1] + 0.5 * diameter]
    c = [i + 0.5 * diameter for i in centroid]
    d = [centroid[0] + 0.5 * diameter, centroid[1] - 0.5 * diameter]
    delta = diameter / num
    ab = {
        'x': np.around([a[0] for _ in range(num)], decimals=decimal),
        'y': np.around([a[1] + i * delta for i in range(num)], decimals=decimal)
    }
    bc = {
        'x': np.around([b[0] + i * delta for i in range(num)], decimals=decimal),
        'y': np.around([b[1] for _ in range(num)], decimals=decimal)
    }
    cd = {
        'x': np.around([c[0] for _ in range(num)], decimals=decimal),
        'y': np.around([c[1] - i * delta for i in range(num)], decimals=decimal)
    }
    da = {
        'x': np.around([d[0] - i * delta for i in range(num)], decimals=decimal),
        'y': np.around([d[1] for _ in range(num)], decimals=decimal)
    }

    cirprobe = {
        'ab': ab,
        'bc': bc,
        'cd': cd,
        'da': da
    }
    # print(cirprobe)

    if saver:
        with open(saver, 'w') as f:
            for i in range(num):
                f.write(f"{' '*indent}({cirprobe['ab']['x'][i]} {cirprobe['ab']['y'][i]} {0})\n")
            for i in range(num):
                f.write(f"{' '*indent}({cirprobe['bc']['x'][i]} {cirprobe['bc']['y'][i]} {0})\n")
            for i in range(num):
                f.write(f"{' '*indent}({cirprobe['cd']['x'][i]} {cirprobe['cd']['y'][i]} {0})\n")
            for i in range(num):
                f.write(f"{' '*indent}({cirprobe['da']['x'][i]} {cirprobe['da']['y'][i]} {0})\n")

    return cirprobe
"""

"""
def wind_pressure_coeffs(path, const, figure=True, saver=False):
    df = read_foam_file(path)
    df_cp = df.apply(lambda x: x-const['pref']/(0.5*const['rho']*const['v']**2))
    df_describe = df_cp.describe()
    if figure:
        df_describe.loc['mean'].plot()
        plt.show()
    if saver:
        df_cp.to_csv(saver)
        df_describe.to_csv(saver)
    return df_cp
"""

"""
def validation(root_path, const, time_figure=True, freq_figure=True, coeffs_figure=True, saver=False):
    forceCoeffs_path = root_path + '/postProcessing/forceCoeffsIncompressible/0/forceCoeffs.dat'
    forceCoeffs_data = read_foam_file(forceCoeffs_path).to_numpy()
    if 'interval' in const:
        assert np.min(const['interval']) >= 0, 'Interval error'
        if np.max(const['interval']) <= 1:
            interval = list(map(int, forceCoeffs_data.shape[0] * np.array(const['interval'])))
        else:
            interval = const['interval']
    else:
        interval = [None, None]
    num_selected_points = forceCoeffs_data[interval[0]:interval[1], 3].shape[0]
    Cl_mean = np.mean(forceCoeffs_data[interval[0]:interval[1], 3])
    Cl_RMS = np.sqrt(np.sum(forceCoeffs_data[interval[0]:interval[1], 3] ** 2) / num_selected_points)
    Cd_mean = np.mean(forceCoeffs_data[interval[0]:interval[1], 2])


    Cl = forceCoeffs_data[interval[0]:interval[1], 3]
    t_s = forceCoeffs_data[-1, 0] - forceCoeffs_data[-2, 0]
    f_s = 1 / t_s
    F = np.fft.fft(Cl)
    f = np.fft.fftfreq(num_selected_points, t_s)
    mask = np.where(f >= 0)

    if 'threshold' in const:
        threshold = const['threshold']
    else:
        threshold = 0.3
    if 'min_distance' in const:
        min_distance = const['min_distance']
    else:
        min_distance = 1
    peaks_index = peakutils.indexes(np.abs(F[mask])/num_selected_points, thres=threshold, min_dist=min_distance)
    peaks_x = np.array(f[peaks_index])
    peaks_y = np.array(np.abs(F[peaks_index])/num_selected_points)
    shedding_frequency = peaks_x[np.argmax(peaks_y)]
    strouhal_number = shedding_frequency * const['D'] / const['U0']
    # print(np.around(shedding_frequency, decimals=3))

    if coeffs_figure:
        if 'p0' in const:
            p0 = const['p0']
        else:
            p0 = 0
        if 'rho' in const:
            rho = const['rho']
        else:
            rho = 1.225
        p_path = root_path + '/postProcessing/probes/0/p'
        probe_data = read_foam_file(p_path).to_numpy()
        Cp = (probe_data[interval[0]:interval[1], 1:] - p0) / (0.5*rho*const['U0']*const['U0'])
        p_mean = np.mean(Cp, axis=0)

    result = pd.DataFrame({
        'Cl_mean': Cl_mean,
        'Cl_RMS': Cl_RMS,
        'Cd_mean': Cd_mean,
        'num_selected_points': num_selected_points,
        'num_all_points': forceCoeffs_data.shape[0],
        'sampling_frequency': f_s,
        'sampling_period': t_s,
        'shedding_frequency': shedding_frequency,
        'strouhal_number': strouhal_number
    }, index=[0])
    print(result)

    if time_figure:
        Time = forceCoeffs_data[interval[0]:interval[1], 0]
        Cd = forceCoeffs_data[interval[0]:interval[1], 2]
        Cl = forceCoeffs_data[interval[0]:interval[1], 3]

        fig0, axes0 = plt.subplots(2, 1)
        fig0.suptitle(f'{root_path}')
        axes0[0].plot(Time, Cd, label='Cd')
        axes0[0].plot(Time, [Cd_mean for _ in range(len(Time))], 'r--', label=f'mean = {np.around(Cd_mean, decimals=3)}')
        axes0[0].set_ylabel('Cd')
        axes0[0].legend()

        axes0[1].plot(Time, Cl, label='Cl')
        axes0[1].plot(Time, [Cl_mean for _ in range(len(Time))], 'r--', label=f'mean = {np.around(Cl_mean, decimals=3)}')
        axes0[1].plot(Time, [Cl_RMS for _ in range(len(Time))], 'g--', label=f'RMS = {np.around(Cl_RMS, decimals=3)}')
        # axes0[1].set_xlabel('Time (s) / Step')
        axes0[1].set_xlabel('Time (s) / Step')
        axes0[1].set_ylabel('Cl')
        axes0[1].legend()

    if freq_figure:
        fig1, axes1 = plt.subplots(2, 1)
        axes1[0].plot(f[mask], np.abs(F[mask])/num_selected_points, label='Amplitude')
        axes1[0].plot(peaks_x, peaks_y, 'r*', label='peaks')
        axes1[0].plot([shedding_frequency, shedding_frequency], [0, peaks_y.max()*1.1], 'r--', label=f'shedding_frequency')
        delta_f = peaks_x.max() - peaks_x.min()
        axes1[0].set_xlim(peaks_x.min() - delta_f, peaks_x.max() + delta_f)
        for x, y in zip(peaks_x, peaks_y):
            axes1[0].text(x, y, f'{np.around(x, decimals=2)}')
        # axes1[0].set_xlabel('frequency (Hz)')
        axes1[0].set_ylabel('Amp. Spectrum')
        axes1[0].set_title(f'shedding_frequency = {np.around(shedding_frequency, decimals=2)}  strouhal_number = {np.around(strouhal_number, decimals=2)}')
        axes1[0].legend()

        axes1[1].plot(f[mask], np.abs(F[mask])/num_selected_points, label='Amplitude')
        axes1[1].plot(peaks_x, peaks_y, 'r*', label='peaks')
        axes1[1].set_xlabel('frequency (Hz)')
        axes1[1].set_ylabel('Amp. Spectrum')
        axes1[1].legend()

    if coeffs_figure:
        with open(TensorFOAM_path + '/Database/Validation/Cp.JSON', 'r') as f:
            Cp_validation_data = json.load(f)

        fig2, axes2 = plt.subplots(2, 1)
        # axes[1].plot(Cp_validation_data['Jiang'][::2], Cp_validation_data['Jiang'][1::2], label='Jiang(CFD)')
        axes2[0].plot(Cp_validation_data['Lee(EXP)'][::2], Cp_validation_data['Lee(EXP)'][1::2], label='Lee(EXP)')
        axes2[0].plot(Cp_validation_data['Bearman(EXP)'][::2], Cp_validation_data['Bearman(EXP)'][1::2], label='Bearman & Obasaju(EXP)')
        axes2[0].plot(np.linspace(0, 4, len(p_mean)), p_mean, 'k', label='Case')
        axes2[0].set_xticks([0, 1, 2, 3, 4])
        axes2[0].set_xticklabels(['A', 'B', 'C', 'D', 'A'])
        axes2[0].legend()

        # interpolation
        f1 = interp1d(Cp_validation_data['Lee(EXP)'][::2], Cp_validation_data['Lee(EXP)'][1::2])
        f2 = interp1d(Cp_validation_data['Bearman(EXP)'][::2], Cp_validation_data['Bearman(EXP)'][1::2])
        x_min = np.max([np.min(Cp_validation_data['Lee(EXP)'][::2]), np.min(Cp_validation_data['Bearman(EXP)'][::2])])
        x_max = np.min([np.max(Cp_validation_data['Lee(EXP)'][::2]), np.max(Cp_validation_data['Bearman(EXP)'][::2])])
        x_new = np.linspace(x_min, x_max, 100)
        f = (f1(x_new) + f2(x_new)) / 2
        axes2[1].plot(x_new, f, label='average')
        axes2[1].plot(np.linspace(0, 4, len(p_mean)), p_mean, 'k', label='Case')
        axes2[1].legend()

    if time_figure or freq_figure:
        plt.tight_layout()
        plt.show()

    if saver:
        result.to_csv(saver)

    return result
"""

# def parse_init(str_exprs, input_var):
#     """
#     exprs = [
#         'x + y + z - 1',
#         'x + y + 2*z - 3'
#     ]
#     """
#     var = set()
#     for expr in str_exprs:
#         sym_exprs = sympy.sympify(expr)
#         var = var.union(sym_exprs.free_symbols)
#     all_var = tuple(var)
#     assert ({str(i) for i in var} == set(input_var)), 'Input variables do not correspond to constrained equations.'
#     sym_solution = sympy.linsolve(str_exprs, all_var)
#     independent_var = tuple(sym_solution.free_symbols)
#     return all_var, sym_solution.args[0], independent_var
#
#
# def parse(init_parse, actions):
#     if isinstance(actions, (int, float)):
#         actions = [actions]
#     var_dict = dict(zip(init_parse[2], actions))
#     all_sym_value = init_parse[1].subs(var_dict)
#     all_str_var = tuple(str(i) for i in init_parse[0])
#     all_float_value = {float(i) for i in all_sym_value}
#     return dict(zip(all_str_var, all_float_value))

"""
def get_history_data(dir_path, dimension=3):
    time_list = [i for i in os.listdir(dir_path)]
    time_list_to_num = [np.float(i) for i in time_list]
    time_index_minmax = np.argsort(time_list_to_num)
    file_name = os.listdir(dir_path + f'/{time_list[0]}')
    count = 0
    dataframe_history_data = 0
    for index in time_index_minmax:
        count = count + 1
        current_time_path = dir_path + f'/{time_list[index]}/{file_name[0]}'
        # print(current_time_path, count)
        if count < 1.5:
            dataframe_history_data = read_foam_file(current_time_path, dimension=dimension)
        else:
            dataframe_history_data = pd.concat([dataframe_history_data, read_foam_file(current_time_path, dimension=dimension)[1:]]).reset_index(drop=True)
    dataframe_history_data.to_csv('./data.csv', index=False)
    # dataframe_history_data.to_csv('./data.csv', index=False, header=False)
"""

def resultant_force(dataframe, saver=False):
    Time = dataframe.iloc[:, 0]
    Fp = dataframe.iloc[:, [1, 2, 3]]
    Fp.columns = ['FX', 'FY', 'FZ']
    Fv = dataframe.iloc[:, [4, 5, 6]]
    Fv.columns = ['FX', 'FY', 'FZ']
    Fo = dataframe.iloc[:, [7, 8, 9]]
    Fo.columns = ['FX', 'FY', 'FZ']
    Mp = dataframe.iloc[:, [10, 11, 12]]
    Mp.columns = ['MX', 'MY', 'MZ']
    Mv = dataframe.iloc[:, [13, 14, 15]]
    Mv.columns = ['MX', 'MY', 'MZ']
    Mo = dataframe.iloc[:, [16, 17, 18]]
    Mo.columns = ['MX', 'MY', 'MZ']

    result = pd.concat([pd.concat([Time, Fp + Fv + Fo], axis=1), Mp + Mv + Mo], axis=1)

    if saver:
        result.to_csv(saver)

    return result

def timeit(params):
    """Record the running time of the function, params passes in the display string"""
    def inner(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            print(f'{params} running time：', np.around(end_time-start_time, decimals=2), 's')
        return wrapper
    return inner

def read_foam_file(path, mandatory=False, saver=False, dimension=3):
    """Extract any file including system/probes, postProcessing/.*,
    store results in a DataFrame object with column title,
    and without specify index.

    Parameters
    ----------
    path : str
        Path to simulation file.
    mandatory : bool, optional
    saver : bool, optional
    dimension : int
        Running dimension.

    Returns
    -------
    data_frame_obj
        DataFrame object.

    Examples
    --------
    from DRLinFluids.utils import read_foam_file

    note
    --------
    When reading the forces file, the header is defined as:
    Fp: sum of forces induced by pressure
    Fv: sum of forces induced by viscous
    Fo: sum of forces induced by porous
    Mp: sum of moments induced by pressure
    Mv: sum of moments induced by viscous
    Mo: sum of moments induced by porous
    """
    # Determine whether to read the system/probe file
    if path.split('/')[-2] == 'system':
        if path.split('/')[-1] == 'probes':
            with open(path, 'r') as f:
                content_total = f.read()
                right_str = re.sub('\);?', '', re.sub('[ \t]*\(', '', content_total))
                annotation_num = 0
            for line in right_str.split('\n'):
                if re.search('^-?\d+', line):
                    break
                annotation_num += 1
            right_content = StringIO(right_str)
            data_frame_obj = pd.read_csv(right_content, sep=' ', skiprows=annotation_num, header=None,
                                         names=['x', 'y', 'z'])
        else:
            data_frame_obj = False
            assert data_frame_obj, f'Unknown system/file type\n{path}'
    # Determine whether to read postProcessing/* files
    elif path.split('/')[-4] == 'postProcessing':
        # Write the postProcess file to the variable content_total and count the number of comment lines annotation_num
        with open(path, 'r') as f:
            content_total = f.read()
            f.seek(0)
            content_lines = f.readlines()
            annotation_num = 0
            for line in content_lines:
                if line[0] == '#':
                    annotation_num += 1
                else:
                    break
        if path.split('/')[-1] == 'forces.dat':
            column_name = ['Time']
            column_name.extend(['Fpx', 'Fpy', 'Fpz'])
            column_name.extend(['Fvx', 'Fvy', 'Fvz'])
            column_name.extend(['Fox', 'Foy', 'Foz'])
            column_name.extend(['Mpx', 'Mpy', 'Mpz'])
            column_name.extend(['Mvx', 'Mvy', 'Mvz'])
            column_name.extend(['Mox', 'Moy', 'Moz'])
            right_content = StringIO(re.sub('\)', '', re.sub('\(', '', re.sub('\t+', '\t', re.sub(' +', '\t',
                                                                                                  re.sub('# ', '',
                                                                                                         re.sub(
                                                                                                             '[ \t]+\n',
                                                                                                             '\n',
                                                                                                             content_total)))))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False,
                                         names=column_name)
        elif path.split('/')[-1] == 'p':
            right_content = StringIO(
                re.sub('\t\n', '\n', re.sub(' +', '\t', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num - 1, index_col=False)
        elif path.split('/')[-1] == 'U':
            column_name = ['Time']
            for n in range(annotation_num - 1):
                column_name.append(f'Ux_{n}')
                column_name.append(f'Uy_{n}')
                column_name.append(f'Uz_{n}')
            # print(len(column_name))
            right_content = StringIO(
                re.sub(' +', '\t', re.sub('[\(\)]', '', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False,
                                         names=column_name)
            if dimension == 2:
                drop_column = [i for i in column_name if re.search('^Uz_\d', i)]
                data_frame_obj.drop(drop_column, axis=1, inplace=True)
        elif path.split('/')[-1] == 'forceCoeffs.dat':
            column_name = ['Time', 'Cm', 'Cd', 'Cl', 'Cl(f)', 'Cl(r)']
            right_content = StringIO(re.sub('[ \t]+', '\t', re.sub('[ \t]+\n', '\n', content_total)))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False,
                                         names=column_name)
        # If they do not match, return an error directly
        else:
            if mandatory:
                right_content = StringIO(re.sub(' ', '', re.sub('# ', '', content_total)))
                data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None)
            else:
                data_frame_obj = -1
                assert 0, f'Unknown file type, you can force function to read it by using \'mandatory\' parameters (scalar-like data structure)\n{path}'
    else:
        data_frame_obj = -1
        assert 0, f'Unknown folder path\n{path}'

    if saver:
        data_frame_obj.to_csv(saver, index=False, header=False)

    return data_frame_obj

def actions2dict(entry_dict, reinforcement_learning_var, agent_actions):
    """Write parameters to velocity or pressure files in OpenFOAM.

    Parameters
    ----------
    entry_dict : list
        list for change item.
    reinforcement_learning_var : tuple
        reinforcement learning variables
    agent_actions : tuple
        The action value generated by the agent.

    Returns
    -------
    actions_dict
        Specify reinforcement learning variables.

    Examples
    --------
    from DRLinFluids.utils import actions2dict
    reinforcement_learning_var_example = (x, y, z)
    agent_actions_example = (1, 2, 3)

    note
    --------
    In particular, entry_dict should not contain regular expression related,
    minmax_value: Specify independent variables and expressions related to boundary conditions.
    Note that only the most basic calculation expressions are accepted, and operation functions
    in math or numpy cannot be used temporarily.
    entry_example = {
        'U': {
            'JET1': '({x} 0 0)',
            'JET2': '(0 {y} 0)',
            'JET3': '(0 0 {z})',
            'JET4': '{x+y+z}',
            'JET(4|5)': '{x+y+z}'  # X cannot contain regular expressions and must be separated manually
        },
        'k': {
            'JET1': '({0.5*x} 0 0)',
            'JET2': '(0 {0.5*y*y} 0)',
            'JET3': '(0 0 {2*z**0.5})',
            'JET4': '{x+2*y+3*z}'
        }
    }
    """
    mapping_dict = dict(zip(reinforcement_learning_var, agent_actions))

    entry_str_org = json.dumps(entry_dict, indent=4)
    entry_str_temp = re.sub(r'{\n', r'{{\n', entry_str_org)
    entry_str_temp = re.sub(r'}\n', r'}}\n', entry_str_temp)
    entry_str_temp = re.sub(r'}$', r'}}', entry_str_temp)
    entry_str_temp = re.sub(r'},', r'}},', entry_str_temp)
    entry_str_final = eval(f'f"""{entry_str_temp}"""', mapping_dict)
    actions_dict = json.loads(entry_str_final)

    return actions_dict

def dict2foam(flow_var_directory, actions_dict):
    """Write parameters to velocity or pressure files in OpenFOAM.

    Parameters
    ----------
    flow_var_directory : str
        Path to simulation file.
    actions_dict : tuple
        Specify reinforcement learning variables

    Examples
    --------
    from DRLinFluids.utils import dict2foam
    """
    for flow_var, flow_dict in actions_dict.items():
        with open('/'.join([flow_var_directory, flow_var]), 'r+') as f:
            content = f.read()
            for entry, value_dict in flow_dict.items():
                for keyword, value in value_dict.items():
                    content = re.sub(f'({entry}(?:\n.*?)*{keyword}\s+).*;', f'\g<1>{value};', content)
                # content = re.sub(f'({entry}(?:\n.*?)*value\s+uniform\s+).*;', f'\g<1>{value};', content)
            f.seek(0)
            f.truncate()
            f.write(content)

def get_current_time_path(path):
    """Enter the root directory of the OpenFOAM study, the return value is
    (current (maximum) time folder name|absolute path).

    Parameters
    ----------
    path : str
        Path to simulation file.

    Returns
    -------
    start_time_str
        Current latest time folder name.
    start_time_path
        Current latest time folder path.

    Examples
    --------
    from DRLinFluids.utils import get_current_time_path
    """
    time_list = [i for i in os.listdir(path)]
    temp_list = time_list
    for i, value in enumerate(time_list):
        if re.search('^\d+\.?\d*', value):
            pass
        else:
            temp_list[i] = -1
    time_list_to_num = [np.float(i) for i in temp_list]
    start_time_str = time_list[np.argmax(time_list_to_num)]
    start_time_path = '/'.join([path, start_time_str])

    return start_time_str, start_time_path


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default=os.getcwd(), type=str, help='Working directory')
    parser.add_argument('-f', '--func1', type=str, help='choose utilities')
    shell_args = vars(parser.parse_args())

    # check_ports('127.0.0.1', [9001, 9002])
    # resultant_force(read_foam_file(os.getcwd() + '/env/postProcessing/forcesIncompressible/5.4/forces.dat'), saver='rf.csv')
    # best_training_episode(os.getcwd())
    # yield_cirprobe(1, 25, saver=os.getcwd()+'/probe.txt', indent=4)
    # yield_cirprobe(1, 10, saver=os.getcwd()+'/cir.csv', indent=12)

    const = {
        'D': 0.1,
        'U0': 1,
        'interval': [0.5, 1],
        'threshold': 0.25
    }
    validation(shell_args['path'], const=const, coeffs_figure=False)

    # validation('/home/data/userdata3/Fusob/DRL_case/01', const=const)
    # print(read_foam_file('/home/data/userdata3/Fusob/verify/Final/JET8_RL/env04/postProcessing/probes/0.05405/U'))

    # get_history_data(shell_args['path'], dimension=3)
"""
