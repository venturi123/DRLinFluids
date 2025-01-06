import os
import re
from io import StringIO

import numpy as np
import pandas as pd


def read_foam_file(path, mandatory=False, saver=False, dimension=3):
    """Read OpenFOAM result file, such as system/probes and postProcessing/* files.

    Returns
    -------
    pandas.DataFrame
        _description_
    """
    if path.split("/")[-2] == "system":
        if path.split("/")[-1] == "probes":
            with open(path, "r") as f:
                content_total = f.read()
                right_str = re.sub("\);?", "", re.sub("[ \t]*\(", "", content_total))
                annotation_num = 0
            for line in right_str.split("\n"):
                if re.search("^-?\d+", line):
                    break
                annotation_num += 1
            right_content = StringIO(right_str)
            data_frame_obj = pd.read_csv(
                right_content,
                sep=" ",
                skiprows=annotation_num,
                header=None,
                names=["x", "y", "z"],
            )
        else:
            data_frame_obj = False
            assert data_frame_obj, f"Unknown system/file type\n{path}"

    elif path.split("/")[-4] == "postProcessing":
        with open(path, "r") as f:
            content_total = f.read()
            f.seek(0)
            content_lines = f.readlines()
            annotation_num = 0
            for line in content_lines:
                if line[0] == "#":
                    annotation_num += 1
                else:
                    break
        if path.split("/")[-1] == "forces.dat":
            column_name = ["Time"]
            column_name.extend(["Fpx", "Fpy", "Fpz"])
            column_name.extend(["Fvx", "Fvy", "Fvz"])
            column_name.extend(["Fox", "Foy", "Foz"])
            column_name.extend(["Mpx", "Mpy", "Mpz"])
            column_name.extend(["Mvx", "Mvy", "Mvz"])
            column_name.extend(["Mox", "Moy", "Moz"])
            right_content = StringIO(
                re.sub(
                    "\)",
                    "",
                    re.sub(
                        "\(",
                        "",
                        re.sub(
                            "\t+",
                            "\t",
                            re.sub(
                                " +",
                                "\t",
                                re.sub(
                                    "# ", "", re.sub("[ \t]+\n", "\n", content_total)
                                ),
                            ),
                        ),
                    ),
                )
            )
            data_frame_obj = pd.read_csv(
                right_content,
                sep="\t",
                skiprows=annotation_num,
                header=None,
                index_col=False,
                names=column_name,
            )
        elif path.split("/")[-1] == "p":
            right_content = StringIO(
                re.sub(
                    "\t\n",
                    "\n",
                    re.sub(
                        " +",
                        "\t",
                        re.sub("# ", "", re.sub("[ \t]+\n", "\n", content_total)),
                    ),
                )
            )
            data_frame_obj = pd.read_csv(
                right_content, sep="\t", skiprows=annotation_num - 1, index_col=False
            )
        elif path.split("/")[-1] == "U":
            column_name = ["Time"]
            for n in range(annotation_num - 1):
                column_name.append(f"Ux_{n}")
                column_name.append(f"Uy_{n}")
                column_name.append(f"Uz_{n}")
            right_content = StringIO(
                re.sub(
                    " +",
                    "\t",
                    re.sub(
                        "[\(\)]",
                        "",
                        re.sub("# ", "", re.sub("[ \t]+\n", "\n", content_total)),
                    ),
                )
            )
            data_frame_obj = pd.read_csv(
                right_content,
                sep="\t",
                skiprows=annotation_num,
                header=None,
                index_col=False,
                names=column_name,
            )
            if dimension == 2:
                drop_column = [i for i in column_name if re.search("^Uz_\d", i)]
                data_frame_obj.drop(drop_column, axis=1, inplace=True)
        elif path.split("/")[-1] == "forceCoeffs.dat":
            column_name = ["Time", "Cm", "Cd", "Cl", "Cl(f)", "Cl(r)"]
            right_content = StringIO(
                re.sub("[ \t]+", "\t", re.sub("[ \t]+\n", "\n", content_total))
            )
            data_frame_obj = pd.read_csv(
                right_content,
                sep="\t",
                skiprows=annotation_num,
                header=None,
                index_col=False,
                names=column_name,
            )
        else:
            if mandatory:
                right_content = StringIO(
                    re.sub(" ", "", re.sub("# ", "", content_total))
                )
                data_frame_obj = pd.read_csv(
                    right_content, sep="\t", skiprows=annotation_num, header=None
                )
            else:
                data_frame_obj = -1
                assert 0, f"Unknown file type, you can force function to read it by using 'mandatory' parameters (scalar-like data structure)\n{path}"
    else:
        data_frame_obj = -1
        assert 0, f"Unknown folder path\n{path}"

    if saver:
        data_frame_obj.to_csv(saver, index=False, header=False)

    return data_frame_obj


def get_history_data(dir_path, saver=False, dimension=3):
    time_list = [i for i in os.listdir(dir_path)]
    time_list_to_num = [np.float(i) for i in time_list]
    time_index_minmax = np.argsort(time_list_to_num)

    file_name = os.listdir(dir_path + f"/{time_list[0]}")

    count = 0
    dataframe_history_data = 0
    for index in time_index_minmax:
        count = count + 1
        current_time_path = dir_path + f"/{time_list[index]}/{file_name[0]}"

        if count < 1.5:
            dataframe_history_data = read_foam_file(
                current_time_path, dimension=dimension
            )
        else:
            dataframe_history_data = pd.concat(
                [
                    dataframe_history_data,
                    read_foam_file(current_time_path, dimension=dimension)[1:],
                ]
            ).reset_index(drop=True)
    if saver:
        dataframe_history_data.to_csv(saver, index=False, header=False)

    return dataframe_history_data
