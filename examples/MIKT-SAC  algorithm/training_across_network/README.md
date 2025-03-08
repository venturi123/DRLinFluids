# Vanilla example

When going 'out of' a docker there is just a bit of added subtlety, i.e. one has to share the network stack between the docker and the host, and change a few names in the RemoteEnvironment file. In practice:

- on the simulations machine(s): start simulations as usual. One NEEDS to have run the container using the ```-P --net="host"``` options when calling the ```run``` commando the first time (if necessary, spin up a new container). For this to work, needs to use the ```'localhost``` instead of the ```host = socket.getlocalhost()``` host in the RemoteEnvironmentXX.py files (if necessary, edit the source files). Once all edits are made:

```
fenics@main-5:~/local/Cylinder2DFlowControlDRLParallel/ParallelDRLTraining$ python3 launch_servers.py -n 10 -p 3000
```

- on the training machine:

  - on the host, forward the ports. For this, can use the script provided. Need that the machine where launching the script is able to ssh itself to the machine running the trainings (so need to have set up ssh keys and unlocked them; in my case, all is in place, and the IP address used is the one of the main-5 machine from the previous command):

  ```ubuntu@main-6:~/Cylinder2DFlowControlDRLParallel/training_across_network$ bash script_ssh_forward_socket.sh train_10_simulations 3000 10 ubuntu@158.39.75.55```

  - on the docker, launch the training. One NEEDS to have run the container using the ```-P --net="host"``` options when calling the ```run``` commando the first time (if necessary, spin up a new container). For this to work, needs to use the ```'localhost``` instead of the ```host = socket.getlocalhost()``` host in the RemoteEnvironmentXX.py files (if necessary, edit the source files).

  ```fenics@main-6:~/local/Cylinder2DFlowControlDRLParallel/ParallelDRLTraining$ python3 launch_parallel_training.py -p 3000 -n 10```

Then training happens,across the several machines connected through the network.

# More complex training

Of course, by launching several redirection scripts, it is easy to collect trainings from different remote machines, mix trainings locally / remotely, etc. For example:

- on VM1 (16 simulations): ```python3 launch_servers.py -n 16 -p 3000```
- on VM2 (16 simulations): ```python3 launch_servers.py -n 16 -p 3016```
- on VM3 (16 simulations): ```python3 launch_servers.py -n 16 -p 3032```
- on VM4 (16 simulations): ```python3 launch_servers.py -n 16 -p 3048```

- on the VM / host that will run the training:

  - ```bash script_ssh_forward_socket.sh train_4 3000 16 ubuntu@IP_VM1```
  - ```bash script_ssh_forward_socket.sh train_4 3016 16 ubuntu@IP_VM2```
  - ```bash script_ssh_forward_socket.sh train_4 3032 16 ubuntu@IP_VM3```
  - ```bash script_ssh_forward_socket.sh train_4 3048 16 ubuntu@IP_VM4```

- Finally on the VM / docker that will run the training: ```python3 launch_parallel_training.py -n 64 -p 3000```.

This way, the parallel training will communicate with the 4 VMs through the sockets tunneled through ssh, each VM running 16 simulations, i.e., 4 times 16 simulations used in total.
