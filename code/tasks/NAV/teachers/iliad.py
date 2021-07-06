import os
import sys
import random

import agents

from .base import BaseTeacher


class IliadTeacher(BaseTeacher):

    EPS = 1e-6

    def __init__(self, config):

        super(IliadTeacher, self).__init__(config)

        self.describer = agents.load(config, 'describer')
        self.executor  = agents.load(config, 'executor')
        self.random = random.Random(self.config.seed)

    def receive_simulation_data(self, batch):
        self.instructions = [item['instruction'] for item in batch]
        self.equiv_instructions = [item['equiv_instructions'] for item in batch]
        self.gold_paths = [item['path'] for item in batch]

    def describe(self, init_poses, paths):

        # Generate descriptions with pragmatic inference
        pred_instructions, pred_paths, metrics = \
            self.describer.pragmatic_predict(
                init_poses, paths, self.executor, self)

        zipped_info = zip(init_poses, pred_paths, paths)
        for i, (pose, pred_path, gold_path) in enumerate(zipped_info):
            scan = pose[0]
            metric = self.eval(scan, pred_path, gold_path)
            if not metric['score']:
                pred_instructions[i] = ['<PAD>']

        # Score trajectories
        is_satisfied = self.is_satisfied(init_poses, self.instructions, paths)
        num_gt_d_hat = 0

        descriptions = []
        for i in range(len(paths)):
            # For high-quality trajectories, use gold instructions
            if is_satisfied[i]:
                num_gt_d_hat += 1
                d = self.random.choice(self.equiv_instructions[i])
                descriptions.append(d)
            # For low-quality trajectories, use predicted descriptions
            else:
                descriptions.append(pred_instructions[i])

        return descriptions, num_gt_d_hat

    def is_satisfied(self, init_poses, instructions, pred_paths):
        is_satisfied = []
        zipped_info = zip(init_poses, pred_paths, self.gold_paths)
        for pose, pred_path, gold_path in zipped_info:
            scan = pose[0]
            metric = self.eval(scan, pred_path, gold_path)
            is_satisfied.append(metric['sdtw'] >= 0.5)
        return is_satisfied



