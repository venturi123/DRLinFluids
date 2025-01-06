import os
import re
import socket
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peakutils
import sympy
from scipy import signal

from drlinfluids.extractor import *


def test():
    print("Import utils.py sucessfully!")


def freq_domain_analysis(data, const, threshold=0.5, min_distance=30) -> dict:
    """Frequency Domain Analysis of Lift and Drag Results Using Fast Fourier Transform (FFT).

    Parameters
    ----------
    data : pandas.DataFrame
        _description_
    const : _type_
        _description_
    threshold : float, optional
        _description_, by default 0.5
    min_distance : int, optional
        _description_, by default 30

    Returns
    -------
    dict
        _description_
    """
    if "interval" in const:
        interval = const["interval"]
    else:
        interval = [None, None]
    data = np.array(data)
    num_selected_points = data[interval[0] : interval[1]].shape[0]
    cl_mean = np.mean(data[interval[0] : interval[1]])
    cl_rms = np.sqrt(np.sum(data[interval[0] : interval[1]] ** 2) / num_selected_points)
    cd_mean = np.mean(data[interval[0] : interval[1], 2])

    Cl = data[interval[0] : interval[1]]
    t_s = data[1, 0] - data[0, 0]
    f_s = 1 / t_s

    F = np.fft.fft(Cl)
    f = np.fft.fftfreq(num_selected_points, t_s)
    mask = np.where(f >= 0)

    peaks_index = peakutils.indexes(
        np.abs(F[mask]) / num_selected_points, thres=threshold, min_dist=min_distance
    )
    peaks_x = np.array(f[peaks_index])
    peaks_y = np.array(np.abs(F[peaks_index]) / num_selected_points)

    shedding_frequency = np.sum(peaks_x * peaks_y) / peaks_y.sum()
    strouhal_number = shedding_frequency * const["D"] / const["U"]

    result = {
        "Cl_mean": cl_mean,
        "Cl_RMS": cl_rms,
        "Cd_mean": cd_mean,
        "num_selected_points": num_selected_points,
        "num_all_points": data.shape[0],
        "sampling_frequency": f_s,
        "sampling_period": t_s,
        "shedding_frequency": shedding_frequency,
        "strouhal_number": strouhal_number,
    }

    return result


def resultant_force(dataframe, saver=False):
    """_summary_

    Parameters
    ----------
    dataframe : _type_
        _description_
    saver : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    Time = dataframe.iloc[:, 0]
    Fp = dataframe.iloc[:, [1, 2, 3]]
    Fp.columns = ["FX", "FY", "FZ"]
    Fv = dataframe.iloc[:, [4, 5, 6]]
    Fv.columns = ["FX", "FY", "FZ"]
    Fo = dataframe.iloc[:, [7, 8, 9]]
    Fo.columns = ["FX", "FY", "FZ"]
    Mp = dataframe.iloc[:, [10, 11, 12]]
    Mp.columns = ["MX", "MY", "MZ"]
    Mv = dataframe.iloc[:, [13, 14, 15]]
    Mv.columns = ["MX", "MY", "MZ"]
    Mo = dataframe.iloc[:, [16, 17, 18]]
    Mo.columns = ["MX", "MY", "MZ"]

    result = pd.concat([pd.concat([Time, Fp + Fv + Fo], axis=1), Mp + Mv + Mo], axis=1)

    if saver:
        result.to_csv(saver)

    return result


def timeit(params):
    """_summary_

    Parameters
    ----------
    params : _type_
        _description_
    """

    def inner(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            print(
                f"{params} running time：",
                np.around(end_time - start_time, decimals=2),
                "s",
            )

        return wrapper

    return inner


def check_ports(host, port, num_ports=0, verbose=True):
    """_summary_

    Parameters
    ----------
    host : _type_
        _description_
    port : _type_
        _description_
    num_ports : int, optional
        _description_, by default 0
    verbose : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    if isinstance(port, int) and not num_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                if verbose:
                    print(f"host {host} on port {port} is AVAILABLE")
                return True
            except:
                if verbose:
                    print(f"host {host} on port {port} is BUSY")
                return False
    elif isinstance(port, list) and not num_ports:
        for crrt_port in port:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((host, crrt_port))
                    if verbose:
                        print(f"host {host} on port {crrt_port} is AVAILABLE")
                except:
                    if verbose:
                        print(f"host {host} on port {crrt_port} is BUSY")
                    return False
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
                        print(f"host {host} on port {crrt_port} is AVAILABLE")
                except:
                    if verbose:
                        print(f"host {host} on port {crrt_port} is BUSY")
                    return False
        if verbose:
            print("all ports available")
        return True
    else:
        assert 0, "check input arguments!"


def yield_cirprobe(
    diameter, num, decimal=6, centroid=(0, 0), saver=False, indent=4
) -> dict:
    """_summary_

    Parameters
    ----------
    diameter : _type_
        _description_
    num : _type_
        _description_
    decimal : int, optional
        _description_, by default 6
    centroid : tuple, optional
        _description_, by default (0, 0)
    saver : bool, optional
        _description_, by default False
    indent : int, optional
        _description_, by default 4

    Returns
    -------
    dict
        _description_
    """
    a = [i - 0.5 * diameter for i in centroid]
    b = [centroid[0] - 0.5 * diameter, centroid[1] + 0.5 * diameter]
    c = [i + 0.5 * diameter for i in centroid]
    d = [centroid[0] + 0.5 * diameter, centroid[1] - 0.5 * diameter]
    delta = diameter / num
    ab = {
        "x": np.around([a[0] for _ in range(num)], decimals=decimal),
        "y": np.around([a[1] + i * delta for i in range(num)], decimals=decimal),
    }
    bc = {
        "x": np.around([b[0] + i * delta for i in range(num)], decimals=decimal),
        "y": np.around([b[1] for _ in range(num)], decimals=decimal),
    }
    cd = {
        "x": np.around([c[0] for _ in range(num)], decimals=decimal),
        "y": np.around([c[1] - i * delta for i in range(num)], decimals=decimal),
    }
    da = {
        "x": np.around([d[0] - i * delta for i in range(num)], decimals=decimal),
        "y": np.around([d[1] for _ in range(num)], decimals=decimal),
    }

    cirprobe = {"ab": ab, "bc": bc, "cd": cd, "da": da}
    # print(cirprobe)

    if saver:
        with open(saver, "w") as f:
            for i in range(num):
                f.write(
                    f"{' ' * indent}({cirprobe['ab']['x'][i]} {cirprobe['ab']['y'][i]} {0})\n"
                )
            for i in range(num):
                f.write(
                    f"{' ' * indent}({cirprobe['bc']['x'][i]} {cirprobe['bc']['y'][i]} {0})\n"
                )
            for i in range(num):
                f.write(
                    f"{' ' * indent}({cirprobe['cd']['x'][i]} {cirprobe['cd']['y'][i]} {0})\n"
                )
            for i in range(num):
                f.write(
                    f"{' ' * indent}({cirprobe['da']['x'][i]} {cirprobe['da']['y'][i]} {0})\n"
                )

    return cirprobe


def wind_pressure_coeffs(path, const, figure=True, saver=False):
    """_summary_

    Parameters
    ----------
    path : _type_
        _description_
    const : _type_
        _description_
    figure : bool, optional
        _description_, by default True
    saver : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    df = read_foam_file(path)
    df_cp = df.apply(
        lambda x: x - const["pref"] / (0.5 * const["rho"] * const["v"] ** 2)
    )
    df_describe = df_cp.describe()
    if figure:
        df_describe.loc["mean"].plot()
        plt.show()
    if saver:
        df_cp.to_csv(saver)
        df_describe.to_csv(saver)
    return df_cp


def parse_init(str_exprs, input_var):
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
    var = set()
    for expr in str_exprs:
        sym_exprs = sympy.sympify(expr)
        var = var.union(sym_exprs.free_symbols)
    all_var = tuple(var)
    assert {str(i) for i in var} == set(
        input_var
    ), "Input variables do not correspond to constrained equations."
    sym_solution = sympy.linsolve(str_exprs, all_var)
    independent_var = tuple(sym_solution.free_symbols)
    return all_var, sym_solution.args[0], independent_var


def parse(init_parse, actions):
    """_summary_

    Parameters
    ----------
    init_parse : _type_
        _description_
    actions : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if isinstance(actions, (int, float)):
        actions = [actions]
    var_dict = dict(zip(init_parse[2], actions))
    all_sym_value = init_parse[1].subs(var_dict)
    all_str_var = tuple(str(i) for i in init_parse[0])
    all_float_value = {float(i) for i in all_sym_value}
    return dict(zip(all_str_var, all_float_value))


def get_current_time_path(foam_root_path):
    """_summary_

    Parameters
    ----------
    foam_root_path : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    time_list = [i for i in os.listdir(foam_root_path)]
    temp_list = time_list
    for i, value in enumerate(time_list):
        if re.search("^\d+\.?\d*", value):
            pass
        else:
            temp_list[i] = -1
    time_list_to_num = [np.float(i) for i in temp_list]
    start_time_str = time_list[np.argmax(time_list_to_num)]
    start_time_path = "/".join([foam_root_path, start_time_str])

    return start_time_str, start_time_path


def force_coeffs_sliding_average(
    history_force_Coeffs: pd.DataFrame, sliding_time_interval: float, delta_t: float
):
    sampling_num = int(sliding_time_interval / delta_t)
    if history_force_Coeffs.shape[0] <= sampling_num:
        sliding_average_cd = np.mean(
            signal.savgol_filter(history_force_Coeffs.iloc[:, 2], 49, 0)
        )
        sliding_average_cl = np.mean(
            signal.savgol_filter(history_force_Coeffs.iloc[:, 3], 49, 0)
        )
    else:
        sliding_average_cd = np.mean(
            signal.savgol_filter(history_force_Coeffs.iloc[-sampling_num:, 2], 49, 0)
        )
        sliding_average_cl = np.mean(
            signal.savgol_filter(history_force_Coeffs.iloc[-sampling_num:, 3], 49, 0)
        )
    return sliding_average_cd, sliding_average_cl


def sliding_history_force_coeffs(
    history_force_Coeffs: pd.DataFrame, sliding_time_interval: float, delta_t: float
):
    sampling_num = int(sliding_time_interval / delta_t)
    if history_force_Coeffs.shape[0] <= sampling_num:
        sliding_history_cd = history_force_Coeffs.iloc[:, 2]
        sliding_history_cl = history_force_Coeffs.iloc[:, 3]
    else:
        sliding_history_cd = history_force_Coeffs.iloc[-sampling_num:, 2]
        sliding_history_cl = history_force_Coeffs.iloc[-sampling_num:, 3]
    return sliding_history_cd.to_numpy(), sliding_history_cl.to_numpy()
