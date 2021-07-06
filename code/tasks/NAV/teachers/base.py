import os
import sys
import math

import worlds


class BaseTeacher(object):

    metrics = ['dist', 'score', 'spl', 'ndtw', 'sdtw']

    def __init__(self, config):

        self.config = config
        self.world = worlds.load(config)
        self.success_radius = config.teacher.success_radius

    def get_metrics(self):
        return self.metrics

    def format_metrics(self, metric_dict):
        metric_strings = []
        metric_strings.append('score %.1f' % (metric_dict['score'] * 100))
        metric_strings.append('spl %.1f' % metric_dict['spl'])
        metric_strings.append('dist %.2f' % metric_dict['dist'])
        metric_strings.append('ndtw %.3f' % metric_dict['ndtw'])
        metric_strings.append('sdtw %.3f' % metric_dict['sdtw'])

        return ', '.join(metric_strings)

    def init_metric_value(self, metric_name):
        if metric_name in ['dist']:
            return 1e9
        if metric_name in ['score', 'spl', 'ndtw', 'sdtw']:
            return -1e9
        raise ValueError('%s is not a valid metric' % metric_name)

    def is_better(self, metric_name, a, b):
        if metric_name in ['dist']:
            return a < b
        if metric_name in ['score', 'spl', 'ndtw', 'sdtw']:
            return a > b
        raise ValueError('%s is not a valid metric' % metric_name)

    def eval(self, scan, pred_path, gold_path):

        pred_goal = pred_path[-1]
        gold_goal = gold_path[-1]

        dist = self.world.get_shortest_distance(scan, pred_goal, gold_goal)
        score = dist <= self.success_radius

        pred_travel_dist = self.world.get_path_length(scan, pred_path) + 1e-6
        gold_travel_dist = self.world.get_path_length(scan, gold_path) + 1e-6

        spl = score * gold_travel_dist / max(pred_travel_dist, gold_travel_dist)

        ndtw = self.compute_ndtw(scan, pred_path, gold_path)
        sdtw = self.compute_sdtw(scan, pred_path, gold_path)

        metric = {
                'dist': dist,
                'score': score,
                'spl': spl,
                'ndtw': ndtw,
                'sdtw': sdtw
            }

        return metric

    def compute_ndtw(self, scan, pred_path, gold_path):
        r = gold_path
        q = pred_path
        c = [[1e9] * (len(q) + 1) for _ in range(len(r) + 1)]
        c[0][0] = 0

        for i in range(1, len(r) + 1):
            for j in range(1, len(q) + 1):
                d = self.world.get_shortest_distance(scan, r[i - 1], q[j - 1])
                c[i][j] = min(c[i - 1][j], c[i][j - 1], c[i - 1][j - 1]) + d

        return math.exp(-c[len(r)][len(q)] / (len(r) * self.success_radius))

    def compute_sdtw(self, scan, pred_path, gold_path):
        d = self.world.get_shortest_distance(scan, pred_path[-1], gold_path[-1])
        if d > self.success_radius:
            return 0
        return self.compute_ndtw(scan, pred_path, gold_path)

