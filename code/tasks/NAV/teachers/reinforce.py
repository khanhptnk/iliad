import os
import sys
import random

from .base import BaseTeacher


class ReinforceTeacher(BaseTeacher):

    def __init__(self, config):

        super(ReinforceTeacher, self).__init__(config)

        self.reward_metric = self.config.teacher.reward_metric

    def receive_simulation_data(self, batch):
        self.gold_paths = [item['path'] for item in batch]

    def score(self, init_poses, pred_paths):

        rewards = []

        zipped_info = zip(init_poses, pred_paths, self.gold_paths)
        for pose, pred_path, gold_path in zipped_info:
            scan = pose[0]
            metric = self.eval(scan, pred_path, gold_path)
            reward = metric[self.reward_metric]
            if self.reward_metric == 'dist':
                shortest_dist = self.world.get_shortest_distance(
                    scan, gold_path[0], gold_path[-1])
                reward = (shortest_dist - reward) / shortest_dist
            rewards.append(reward)

        return rewards



