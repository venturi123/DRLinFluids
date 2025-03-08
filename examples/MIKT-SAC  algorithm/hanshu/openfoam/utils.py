# coding = UTF-8
import json
import os
import re
import socket
import time
from io import StringIO
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peakutils
from scipy.interpolate import interp1d

# 当前文件夹绝对路径
TensorFOAM_path = os.path.dirname(os.path.abspath(__file__))


def test():
    print('Success!')


# 定义一个新的数据结构RingBuffer类，其具有先进先出的特点，可以用来储存数据
class RingBuffer():
    """A 1D ring buffer using numpy arrays"""
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        """adds ndarray x to ring buffer"""
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        """Returns the first-in-first-out data in the ring buffer"""
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]


# 使用FFT计算上一个episode的涡脱周期，取其10%作为下一episode的时长，输入变量为建筑物表面升力时程曲线
def freq_domain_analysis(data, const, threshold=0.5, min_distance=30) -> dict:
    if 'interval' in const:
        interval = const['interval']
    else:
        interval = [None, None]
    # 计算Cl_mean、Cl_RMS、Cd_mean、St并绘制Cd、Cl时间曲线
    data = np.array(data)
    num_selected_points = data[interval[0]:interval[1]].shape[0]
    cl_mean = np.mean(data[interval[0]:interval[1]])
    cl_rms = np.sqrt(np.sum(data[interval[0]:interval[1]] ** 2) / num_selected_points)
    cd_mean = np.mean(data[interval[0]:interval[1], 2])
    # 对Cl进行FFT运算
    Cl = data[interval[0]:interval[1]]
    # 采样周期ts
    t_s = data[1, 0] - data[0, 0]
    # 采样频率fs
    f_s = 1 / t_s
    # fft变换
    F = np.fft.fft(Cl)
    f = np.fft.fftfreq(num_selected_points, t_s)
    mask = np.where(f >= 0)
    # 寻峰
    peaks_index = peakutils.indexes(np.abs(F[mask])/num_selected_points, thres=threshold, min_dist=min_distance)
    peaks_x = np.array(f[peaks_index])
    peaks_y = np.array(np.abs(F[peaks_index])/num_selected_points)
    # 求解脱落频率，幅值作为权重的加权平均作为脱落频率
    shedding_frequency = np.sum(peaks_x * peaks_y) / peaks_y.sum()
    # 计算strouhal_number
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


# 读取OpenFOAM的结果文件，需要理清楚的一个概念是，OF的结果文件不仅
# 包含了state(以probe处的U、p为代表)
# 还包括了reward(以建筑物表面forceCoeffs、drag以及lift等为代表)
# 返回结果是一个DataFrame object
def read_foam_file(path, mandatory=False, saver=False, dimension=3):
    """
    提取包括system/probes、postProcessing/.* 下的任意文件，将结果储存在DataFrame 对象中，具有column title，且不指定index

    读取forces文件时，header定义为：
    Fp: sum of forces induced by pressure
    Fv: sum of forces induced by viscous
    Fo: sum of forces induced by porous
    Mp: sum of moments induced by pressure
    Mv: sum of moments induced by viscous
    Mo: sum of moments induced by porous
    """
    # 判断是否读取system/probe文件
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
            data_frame_obj = pd.read_csv(right_content, sep=' ', skiprows=annotation_num, header=None, names=['x', 'y', 'z'])
        else:
            data_frame_obj = False
            assert data_frame_obj, f'Unknown system/file type\n{path}'
    # 判断是否读取postProcessing/*文件
    elif path.split('/')[-4] == 'postProcessing':
        # 将postProcess文件写入变量content_total，并计算注释行数annotation_num
        with open(path, 'r') as f:
            content_total = f.read()
            f.seek(0.000)
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
            right_content = StringIO(re.sub('\)', '', re.sub('\(', '', re.sub('\t+', '\t', re.sub(' +', '\t', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False, names=column_name)
        elif path.split('/')[-1] == 'p':
            right_content = StringIO(re.sub('\t\n', '\n', re.sub(' +', '\t', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num-1, index_col=False)
        elif path.split('/')[-1] == 'U':
            column_name = ['Time']
            for n in range(annotation_num-1):
                column_name.append(f'Ux_{n}')
                column_name.append(f'Uy_{n}')
                column_name.append(f'Uz_{n}')
            # print(len(column_name))
            right_content = StringIO(re.sub(' +', '\t', re.sub('[\(\)]', '', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False, names=column_name)
            if dimension == 2:
                drop_column = [i for i in column_name if re.search('^Uz_\d', i)]
                data_frame_obj.drop(drop_column, axis=1, inplace=True)
        elif path.split('/')[-1] == 'forceCoeffs.dat':
            column_name = ['Time', 'Cm', 'Cd', 'Cl', 'Cl(f)', 'Cl(r)']
            right_content = StringIO(re.sub('[ \t]+', '\t', re.sub('[ \t]+\n', '\n', content_total)))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False, names=column_name)
    # 若均不匹配，直接返回错误
        else:
            if mandatory:
                right_content = StringIO(re.sub(' ', '', re.sub('# ', '', content_total)))
                data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None)
            else:
                data_frame_obj = -1
                assert 0, f'Unknown file type, you can force function to read it by using \'mandatory\' parameters (scalar-like data structure)\n{path}'

    elif path.split('/')[-3] == 'forceCoeffsIncompressible':
        # 将postProcess文件写入变量content_total，并计算注释行数annotation_num
        with open(path, 'r') as f:
            content_total = f.read()
            f.seek(0.000)
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
            right_content = StringIO(re.sub('\)', '', re.sub('\(', '', re.sub('\t+', '\t', re.sub(' +', '\t', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False, names=column_name)
        elif path.split('/')[-1] == 'forceCoeffs.dat':
            column_name = ['Time', 'Cm', 'Cd', 'Cl', 'Cl(f)', 'Cl(r)']
            right_content = StringIO(re.sub('[ \t]+', '\t', re.sub('[ \t]+\n', '\n', content_total)))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False, names=column_name)

    elif path.split('/')[-3] == 'probes':
        # 将postProcess文件写入变量content_total，并计算注释行数annotation_num
        with open(path, 'r') as f:
            content_total = f.read()
            f.seek(0.000)
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
            right_content = StringIO(re.sub('\)', '', re.sub('\(', '', re.sub('\t+', '\t', re.sub(' +', '\t', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False, names=column_name)
        elif path.split('/')[-1] == 'p':
            right_content = StringIO(re.sub('\t\n', '\n', re.sub(' +', '\t', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num-1, index_col=False)
        elif path.split('/')[-1] == 'U':
            column_name = ['Time']
            for n in range(annotation_num-1):
                column_name.append(f'Ux_{n}')
                column_name.append(f'Uy_{n}')
                column_name.append(f'Uz_{n}')
            # print(len(column_name))
            right_content = StringIO(re.sub(' +', '\t', re.sub('[\(\)]', '', re.sub('# ', '', re.sub('[ \t]+\n', '\n', content_total)))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False, names=column_name)
            if dimension == 2:
                drop_column = [i for i in column_name if re.search('^Uz_\d', i)]
                data_frame_obj.drop(drop_column, axis=1, inplace=True)

    # 若均不匹配，直接返回错误
    else:
        data_frame_obj = -1
        assert 0, f'Unknown folder path\n{path}'
    
    if saver:
        data_frame_obj.to_csv(saver, index=False, header=False)

    return data_frame_obj


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


# 记录函数运行时间，params传入显示字符串
def timeit(params):
    def inner(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            print(f'{params} running time：', np.around(end_time-start_time, decimals=2), 's')
        return wrapper
    return inner


# 实际就是一个套壳的for循环，用以检查传入的host是否每一个list_ports都是free状态
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


# 用以读取所有env在training过程中得分最高的episode，并写入文件，需要在实例化class中设定SAVE_BEST_EPISOD设定为True
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


def yield_cirprobe(diameter, num, decimal=6, centroid=(0, 0), saver=False, indent=4) -> dict:
    # 注意saver将传递保存地址
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


def validation(root_path, const, time_figure=True, freq_figure=True, coeffs_figure=True, saver=False):
    forceCoeffs_path = root_path + '/postProcessing/forceCoeffsIncompressible/0/forceCoeffs.dat'
    forceCoeffs_data = read_foam_file(forceCoeffs_path).to_numpy()
    # 能够自动识别绝对范围(> 1)与相对范围(0 < interval < 1)
    if 'interval' in const:
        assert np.min(const['interval']) >= 0, 'Interval error'
        if np.max(const['interval']) <= 1:
            interval = list(map(int, forceCoeffs_data.shape[0] * np.array(const['interval'])))
        else:
            interval = const['interval']
    else:
        interval = [None, None]
    # 计算Cl_mean、Cl_RMS、Cd_mean、St并绘制Cd、Cl时间曲线
    num_selected_points = forceCoeffs_data[interval[0]:interval[1], 3].shape[0]
    Cl_mean = np.mean(forceCoeffs_data[interval[0]:interval[1], 3])
    Cl_RMS = np.sqrt(np.sum(forceCoeffs_data[interval[0]:interval[1], 3] ** 2) / num_selected_points)
    Cd_mean = np.mean(forceCoeffs_data[interval[0]:interval[1], 2])

    # 对Cl进行FFT运算
    Cl = forceCoeffs_data[interval[0]:interval[1], 3]
    # 采样周期ts，注意此处的求解采样周期仅选用两个数据，当使用非均匀采样，即adjustTimeStep时，应十分注意，必须避免这种情况！！！
    t_s = forceCoeffs_data[-1, 0] - forceCoeffs_data[-2, 0]
    # 采样频率fs
    f_s = 1 / t_s
    # fft变换
    F = np.fft.fft(Cl)
    f = np.fft.fftfreq(num_selected_points, t_s)
    mask = np.where(f >= 0)

    # 寻峰，并指定参数
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
    # 将最大值y对应x的频率作为脱落频率
    shedding_frequency = peaks_x[np.argmax(peaks_y)]
    # 计算strouhal_number
    strouhal_number = shedding_frequency * const['D'] / const['U0']
    # print(np.around(shedding_frequency, decimals=3))

    # 计算环建筑物表面风压系数
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
        # 读取JSON中的数据
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


# def parse_init(str_exprs, input_var):
#     """
#     用以解析设定的边界条件限制条件。
#     由于强化学习自身的限制，仅支持等式（组）关系，输入时应将限制式全部移到等号的一边。
#     str_exprs输入形式如下：
#     exprs = [
#         'x + y + z - 1',
#         'x + y + 2*z - 3'
#     ]
#     函数返回值依此为(全部变量|消元后每个变量的表达式|独立变量)，将其放置在init方法中，并赋值self变量永久保存
#     返回tuple中的每个元素都是sympy变量，换句话说，仅仅用于传入parse()计算
#     """
#     var = set()
#     for expr in str_exprs:
#         sym_exprs = sympy.sympify(expr)
#         # 提取所有方程组的变量，包括了独立变量和非独立变量
#         var = var.union(sym_exprs.free_symbols)
#     # 这一步是必须的，因为需要将变量顺序固定下来
#     all_var = tuple(var)
#     # 检验自动获取的变量是否正确
#     assert ({str(i) for i in var} == set(input_var)), 'Input variables do not correspond to constrained equations.'
#     # 符号求解后的方程组，返回FiniteSet代数结果，顺序按照tuple(var)排列
#     sym_solution = sympy.linsolve(str_exprs, all_var)
#     # 提取独立变量
#     independent_var = tuple(sym_solution.free_symbols)
#     return all_var, sym_solution.args[0], independent_var
#
#
# def parse(init_parse, actions):
#     # 如果传入的是单个数值，则套一层list
#     if isinstance(actions, (int, float)):
#         actions = [actions]
#     # 如果此处直接free_symbols会出现顺序混乱的问题，这样解出来的结果也是混乱的，所以必须直接指定顺序，传入一个tuple
#     var_dict = dict(zip(init_parse[2], actions))
#     all_sym_value = init_parse[1].subs(var_dict)
#     all_str_var = tuple(str(i) for i in init_parse[0])
#     all_float_value = {float(i) for i in all_sym_value}
#     # 返回一个字典，key由所有变量组成(str)，value为其对应的数值(float)，至此将所有sympy类变量转化通用类型
#     return dict(zip(all_str_var, all_float_value))


# 根据输入的路径，直接获取所有时间文件夹下的data，并按照时间排列，返回一个包含所有时程数据的pandas变量
def get_history_data(dir_path, dimension=3):
    time_list = [i for i in os.listdir(dir_path)]
    time_list_to_num = [np.float(i) for i in time_list]
    time_index_minmax = np.argsort(time_list_to_num)
    # 随便进一个文件夹获取文件名
    file_name = os.listdir(dir_path + f'/{time_list[0]}')
    # 初始化参数
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


def actions2dict(entry_dict, reinforcement_learning_var, agent_actions):
    """
    特别注意，entry_dict中不应包含正则表达式相关，
    minmax_value: 指定边界条件相关独立变量及表达式，注意只接受最基本计算表达式，暂时不能使用math或numpy中的运算函数
    entry_example = {
        'U': {
            'JET1': '({x} 0 0)',
            'JET2': '(0 {y} 0)',
            'JET3': '(0 0 {z})',
            'JET4': '{x+y+z}',
            'JET(4|5)': '{x+y+z}'  X 不能包含正则表达式写法，必须人为将其分开

        },
        'k': {
            'JET1': '({0.5*x} 0 0)',
            'JET2': '(0 {0.5*y*y} 0)',
            'JET3': '(0 0 {2*z**0.5})',
            'JET4': '{x+2*y+3*z}'
        }
    }

    reinforcement_learning_var: 指定强化学习变量，必须是tuple
    reinforcement_learning_var_example = (x, y, z)

    agent_actions: 从agent传出的raw actions，应为一个tuple
    agent_actions_example = (1, 2, 3)
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
    for flow_var, flow_dict in actions_dict.items():
        with open('/'.join([flow_var_directory, flow_var]), 'r+') as f:
            content = f.read()
            for entry, value_dict in flow_dict.items():
                for keyword, value in value_dict.items():
                    content = re.sub(f'({entry}(?:\n.*?)*{keyword}\s+).*;', f'\g<1>{value};', content)
                # content = re.sub(f'({entry}(?:\n.*?)*value\s+uniform\s+).*;', f'\g<1>{value};', content)
            f.seek(0.000)
            f.truncate()
            f.write(content)


def get_current_time_path(foam_root_path):
    """
    输入OpenFOAM算例根目录，返回值为(当前（最大）时间文件夹名称|绝对路径)
    """
    time_list = [i for i in os.listdir(foam_root_path)]
    temp_list = time_list
    for i, value in enumerate(time_list):
        if re.search('^\d+\.?\d*', value):
            pass
        else:
            temp_list[i] = -1
    time_list_to_num = [np.float(i) for i in temp_list]
    start_time_str = time_list[np.argmax(time_list_to_num)]
    start_time_path = '/'.join([foam_root_path, start_time_str])

    return start_time_str, start_time_path


if __name__ == "__main__":
    # 获取命令行参数
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
    # 计算风压系数时，不同网格要注意调整OpenFOAM中的参考面积
    validation(shell_args['path'], const=const, coeffs_figure=False)

    # validation('/home/data/userdata3/Fusob/DRL_case/01', const=const)
    # print(read_foam_file('/home/data/userdata3/Fusob/verify/Final/JET8_RL/env04/postProcessing/probes/0.05405/U'))

    # get_history_data(shell_args['path'], dimension=3)
