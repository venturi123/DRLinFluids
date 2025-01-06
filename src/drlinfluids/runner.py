import os
import re
import shutil
import subprocess

import drlinfluids.utils as utils


@utils.timeit("OpenFOAM_init")
def run_init(path, foam_params):
    """Run simulation in initial period.

    Parameters
    ----------
    path : str
        Path to simulation file.
    foam_params : list
        Some parameters set before.

    Examples
    --------
    from DRLinFluids.cfd import run_init
    """
    assert foam_params[
        "cfd_init_time"
    ], "\n\nInitialization before training is compulsory!\n"
    control_dict_path = path + "/system/controlDict"
    decompose_par_dict_path = path + "/system/decomposeParDict"

    with open(decompose_par_dict_path, "r+") as f:
        content = f.read()
        content = re.sub(
            "(numberOfSubdomains\s+)\d+;",
            f'\g<1>{foam_params["num_processor"]};',
            content,
        )
        f.seek(0)
        f.truncate()
        f.write(content)

    with open(control_dict_path, "r+") as f:
        content = f.read()
        content = re.sub(
            "(application\s+).+;", f'\g<1>{foam_params["solver"]};', content
        )
        content = re.sub("(deltaT\s+).*;", f'\g<1>{foam_params["delta_t"]};', content)
        content = re.sub("(startFrom\s+).*;", "\g<1>startTime;", content)
        content = re.sub("(startTime\s+).+;", "\g<1>0;", content)
        content = re.sub(
            "(endTime\s+).+;", f'\g<1>{foam_params["cfd_init_time"]};', content
        )
        content = re.sub(
            "(writeInterval\s+).+;", f'\g<1>{foam_params["cfd_init_time"]};', content
        )
        content = re.sub("(purgeWrite\s+).+;", "\g<1>0;", content)
        f.seek(0)
        f.truncate()
        f.write(content)

    if foam_params["verbose"]:
        subprocess.run(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + "decomposePar -force",
            shell=True,
            check=True,
            executable="/bin/bash",
        )
        mpi_process = subprocess.Popen(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel',
            shell=True,
            executable="/bin/bash",
        )

        mpi_process.communicate()
        subprocess.run(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + "reconstructPar",
            shell=True,
            check=True,
            executable="/bin/bash",
        )
    else:
        subprocess.run(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + "decomposePar -force > /dev/null",
            shell=True,
            check=True,
            executable="/bin/bash",
        )
        mpi_process = subprocess.Popen(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel > /dev/null',
            shell=True,
            executable="/bin/bash",
        )

        mpi_process.communicate()
        subprocess.run(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + "reconstructPar > /dev/null",
            shell=True,
            check=True,
            executable="/bin/bash",
        )


def run(path, foam_params, deltaT, start_time, end_time):
    """Run simulation in an agent interaction period.

    Parameters
    ----------
    path : str
        Path to simulation file.
    foam_params : list
        Some parameters set before.
    deltaT : float
        Simulation time step.
    start_time : float
        Start time in an agent interaction period.
    end_time : float
        End time in an agent interaction period.

    Examples
    --------
    from DRLinFluids.cfd import run
    """
    control_dict_path = path + "/system/controlDict"
    decompose_par_dict_path = path + "/system/decomposeParDict"

    assert isinstance(
        end_time, (int, float)
    ), "TypeError: end_time must be int or float type"

    with open(control_dict_path, "r+") as f:
        content = f.read()
        if start_time == "latestTime":
            content = re.sub("(startFrom\s+).*;", "\g<1>latestTime;", content)
        elif isinstance(start_time, (int, float)):
            content = re.sub("(startFrom\s+).*;", "\g<1>startTime;", content)
            content = re.sub("(startTime\s+).+;", f"\g<1>{start_time};", content)
        else:
            assert (
                False
            ), "TypeError: start_time must be int, float or specific strings type"
        content = re.sub("(endTime\s+).*;", f"\g<1>{end_time};", content)
        content = re.sub("(deltaT\s+).*;", f"\g<1>{deltaT};", content)
        f.seek(0)
        f.truncate()
        f.write(content)

    if "num_processor" in foam_params:
        print("num_processor is set")
        with open(decompose_par_dict_path, "r+") as f:
            content = f.read()
            content = re.sub(
                "(numberOfSubdomains\s+).*;",
                f'\g<1>{foam_params["num_processor"]};',
                content,
            )
            f.seek(0)
            f.truncate()
            f.write(content)

    if foam_params["verbose"]:
        subprocess.run(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + "decomposePar -force",
            shell=True,
            check=True,
            executable="/bin/bash",
        )
        mpi_process = subprocess.Popen(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel',
            shell=True,
            executable="/bin/bash",
        )

        mpi_process.communicate()
        subprocess.run(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + "reconstructPar",
            shell=True,
            check=True,
            executable="/bin/bash",
        )
    else:
        subprocess.run(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + "decomposePar -force > /dev/null",
            shell=True,
            check=True,
            executable="/bin/bash",
        )
        mpi_process = subprocess.Popen(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel > /dev/null',
            shell=True,
            executable="/bin/bash",
        )

        mpi_process.communicate()
        subprocess.run(
            f"cd {path}"
            + " && "
            + foam_params["of_env_init"]
            + " && "
            + "reconstructPar > /dev/null",
            shell=True,
            check=True,
            executable="/bin/bash",
        )


def clean(path, params="cleanCase1"):
    """Clean simulation directory.

    Parameters
    ----------
    path : str
        Root path to simulation file.
    params : str
        Clean function.
    """
    dst_path = path + "/.CleanFunctions"
    src_path = os.path.dirname(__file__) + "/utils/CleanFunctions"
    shutil.copyfile(src_path, dst_path)
    subprocess.call(["bash", src_path, params])
    os.remove(dst_path)
