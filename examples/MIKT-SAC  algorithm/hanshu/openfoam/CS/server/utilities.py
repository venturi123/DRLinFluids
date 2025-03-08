# coding = UTF-8
import os
import re
import socket
import time
from io import StringIO

import numpy as np
import pandas as pd
import peakutils


# 定义一个新的数据结构RingBuffer类，其具有先进先出的特点，可以用来储存数据
class RingBuffer():
    'A 1D ring buffer using numpy arrays'
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        'adds ndarray x to ring buffer'
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        'Returns the first-in-first-out data in the ring buffer'
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]




# 使用FFT计算上一个episode的涡脱周期，取其10%作为下一episode的时长，输入变量为建筑物表面升力时程曲线
# TODO 实际数据而言，并不理想，还需要再考虑一下，后续可以将其加入到 max_episode_timesteps(self) 类函数中，返回值为 int 
def next_end_time_fft(data_frame, type, threshold=0.2, min_distance=5):
    colunm_data = data_frame[type]
    N = len(data_frame.index)
    fs = (N-1) / (data_frame.iloc[-1, 0] - data_frame.iloc[1, 0])
    fft_data = np.abs(np.fft.rfft(colunm_data))
    peaks = (0.5 * fs / fft_data.size) * peakutils.indexes(fft_data, thres=threshold, min_dist=min_distance)
    time = 1 / (0.5 * fs / fft_data.size * peaks)
    
    pass



# 读取OpenFOAM的结果文件，需要理清楚的一个概念是，OF的结果文件不仅
# 包含了state(以probe处的U、p为代表)
# 还包括了reward(以建筑物表面forceCoeffs、drag以及lift等为代表)
# 返回结果是一个DataFrame object
def read_foam_file(path, mandatory=False):
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
            data_frame_obj = -1
            assert 0, f'Unknown system/file type\n{path}'
    # 判断是否读取postProcessing/*文件
    elif path.split('/')[-4] == 'postProcessing':
        # 将postProcess文件写入变量content_total，并计算注释行数annotation_num
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
            right_content = StringIO(re.sub('\)', '', re.sub('\(', '', re.sub('\t+', '\t', re.sub(' +', '\t', re.sub('# ', '', content_total))))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False, names=column_name)
        elif path.split('/')[-1] == 'p':
            right_content = StringIO(re.sub('\t\n', '\n', re.sub(' +', '\t', re.sub('# ', '', content_total))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num-1, index_col=False)
        elif path.split('/')[-1] == 'U':
            column_name = ['Time']
            for n in range(annotation_num-1):
                column_name.append(f'Ux_{n}')
                column_name.append(f'Uy_{n}')
                column_name.append(f'Uz_{n}')
            right_content = StringIO(re.sub(' +', '\t', re.sub('[\(\)]', '', re.sub('# ', '', content_total))))
            data_frame_obj = pd.read_csv(right_content, sep='\t', skiprows=annotation_num, header=None, index_col=False, names=column_name)
    # 若均不匹配，直接返回错误
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

    return data_frame_obj


def resultant_force(dataframe, saver=False):
    Time = dataframe.iloc[:,0]
    Fp = dataframe.iloc[:,[1,2,3]]
    Fp.columns = ['FX','FY','FZ']
    Fv = dataframe.iloc[:,[4,5,6]]
    Fv.columns = ['FX','FY','FZ']
    Fo = dataframe.iloc[:,[7,8,9]]
    Fo.columns = ['FX','FY','FZ']
    Mp = dataframe.iloc[:,[10,11,12]]
    Mp.columns = ['MX','MY','MZ']
    Mv = dataframe.iloc[:,[13,14,15]]
    Mv.columns = ['MX','MY','MZ']
    Mo = dataframe.iloc[:,[16,17,18]]
    Mo.columns = ['MX','MY','MZ']

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
            print(f'{params} running time：',np.around(end_time-start_time, decimals=2),'s')
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
                return(True)
            except:
                if verbose:
                    print(f'host {host} on port {port} is BUSY')
                return(False)
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
        return(True)
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


# TODO 计算剩余时间
def remaining_time():

    pass


if __name__ == "__main__":

    check_ports('127.0.0.1', [9001, 9002])
    # resultant_force(read_foam_file(os.getcwd() + '/postProcessing/forcesIncompressible/0/forces.dat'), saver='rf.csv')
