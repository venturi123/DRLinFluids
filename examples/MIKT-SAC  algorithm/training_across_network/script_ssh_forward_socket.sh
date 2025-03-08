#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No arguments provided; -h for help"
    exit 1
fi

if [ "$1" == "-h" ]; then
    echo "A bash script to automatically forward all the needed sockets to remote server through ssh:"
    echo "- create a new tmux session"
    echo "- launch the ssh forwarding commands in it"
    echo "- NOTE: should be launched on the host machine if using a docker"
    echo "- NOTE: need the utils.py to be in the same folder where launched"
    echo "Usage:"
    echo "bash script_ssh_forward_socket.sh session_name first_port num_servers ssh_destination"
    echo "example: bash script_ssh_forward_socket.sh train_1 3000 2 ubuntu@158.39.75.10"
    exit 0

else
    if [ $# -eq 4 ]; then
        # all good
        :
    else
        echo "Wrong number of arguments, abort; see help with -h"
        exit 1
    fi
fi

# check that the tmux session name is free
found_session=$(tmux ls | grep $1 | wc -l)

if [ $found_session != 0 ]; then
    echo "Collision in session name!"
    echo "running:    tmux ls | grep $1"
    echo "returned    $(tmux ls | grep $1)"
    echo "choose a different session name!"
    exit 1
fi

# check that all ports are free
output=$(python3 -c "from utils import bash_check_avail; bash_check_avail($2, $3)")

if [ $output == "T" ]; then
    echo "Ports available, launch..."
else
    if [ $output == "F" ]; then
        echo "Abort; some ports are not avail"
        exit 0
    else
        echo "wrong output checking ports; abort"
        exit 1
    fi
fi


# if I went so far, all ports are free and the tmux is available: can launch!

# create our tmux
tmux new -s $1 -d

let lastPort=$2+$3

for port in $(seq $2 1 $lastPort)
do
    echo "launching redirection for port $port"
    tmux send-keys -t $1 "echo launch redirection for port $port" C-m
    tmux send-keys -t $1 "ssh -N -L $port:127.0.0.1:$port $4 &" C-m
    tmux send-keys -t $1 "echo done" C-m
done

# NOTE:
# to find the redirections and be able to kill them:
# netstat -tpln
