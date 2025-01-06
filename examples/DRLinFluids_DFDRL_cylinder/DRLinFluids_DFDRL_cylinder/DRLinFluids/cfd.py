# coding = UTF-8
import re
import subprocess
from DRLinFluids import utils


def run(
    num_trajectory,
    path,
    foam_params,
    agent_interaction_period,
    purgeWrite_numbers,
    writeInterval,
    deltaT,
    start_time,
    end_time,
):
    control_dict_path = path + "/system/controlDict"
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
        content = re.sub(
            "(writeInterval\s+).*;", f"\g<1>{agent_interaction_period};", content
        )
        content = re.sub("(purgeWrite\s+).*;", f"\g<1>{purgeWrite_numbers};", content)
        content = re.sub("(deltaT\s+).*;", f"\g<1>{deltaT};", content)
        f.seek(0)
        f.truncate()
        f.write(content)

    if foam_params["verbose"]:
        if num_trajectory < 1.5:
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
                + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',
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
        else:
            mpi_process = subprocess.Popen(
                f"cd {path}"
                + " && "
                + foam_params["of_env_init"]
                + " && "
                + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',
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
    else:
        if num_trajectory < 1.5:
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
                + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',
                shell=True,
                executable="/bin/bash",
            )
            mpi_process.communicate()
        else:
            mpi_process = subprocess.Popen(
                f"cd {path}"
                + " && "
                + foam_params["of_env_init"]
                + " && "
                + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',
                shell=True,
                executable="/bin/bash",
            )
            mpi_process.communicate()


@utils.timeit("OpenFOAM_init")
def run_init(path, foam_params):
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
            + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',
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
            + f'mpirun -np {foam_params["num_processor"]} {foam_params["solver"]} -parallel |tee system/log.pimpleFoam > /dev/null',
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
