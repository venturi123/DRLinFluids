import os
import re

import numpy as np
import pandas as pd


def best_training_episode(root_path, saver="/best_training_episode"):
    max_episode_reward = []
    env_name_list = sorted(
        [dir for dir in os.listdir(root_path) if re.search("^env\d+$", dir)]
    )
    env_best_path = [
        "/".join([root_path, dir, "best_episode"]) for dir in env_name_list
    ]
    os.makedirs(root_path + saver)

    for path in env_best_path:
        max_episode_reward.append(
            np.max(pd.read_csv(path + "/total_reward.csv", header=None).to_numpy())
        )

    max_index = max_episode_reward.index(max(max_episode_reward))

    pd.read_csv(env_best_path[max_index] + "/best_actions.csv", header=None).to_csv(
        root_path + saver + "/all_best_action.csv"
    )
    pd.Series(max_episode_reward).to_csv(root_path + saver + "/max_episode_reward.csv")

    with open(root_path + saver + "/info.txt", "w") as f:
        f.write(f"best environment of reward is {max_index}")
