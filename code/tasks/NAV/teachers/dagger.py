import os
import sys
import random

from .base import BaseTeacher


class DaggerTeacher(BaseTeacher):

    STOP = 0

    def __call__(self, states, goal_viewpoints):

        teacher_actions = []
        for state, goal_viewpoint in zip(states, goal_viewpoints):
            teacher_actions.append(
                self._next_action(state, goal_viewpoint))

        return teacher_actions

    def _next_action(self, state, goal_viewpoint):

        curr_viewpoint = state.viewpoint

        if curr_viewpoint == goal_viewpoint:
            return self.STOP

        next_viewpoint = self.world.get_shortest_path(
            state.scan, state.viewpoint, goal_viewpoint)[1]

        for i, loc in enumerate(state.adj_loc_list):
            if loc['nextViewpointId'] == next_viewpoint:
                return i

        return None

    def receive_simulation_data(self, batch):
        self.goals = [item['path'][-1] for item in batch]

    def demonstrate(self, init_poses, instructions, pred_paths):

        batch_size = len(pred_paths)
        gold_action_seqs = [[] for _ in range(batch_size)]

        states = self.world.init(init_poses)
        t = 0
        while True:

            gold_actions = self(states, self.goals)
            pred_actions = [None] * batch_size

            for i, state in enumerate(states):
                if t < len(pred_paths[i]):
                    gold_action_seqs[i].append(gold_actions[i])

                if t + 1 >= len(pred_paths[i]):
                    pred_actions[i] = self.STOP
                else:
                    for j, loc in enumerate(state.adj_loc_list):
                        if loc['nextViewpointId'] == pred_paths[i][t + 1]:
                            pred_actions[i] = j
                            break

            states = states.step(pred_actions)

            t += 1

            if all([a == self.STOP for a in pred_actions]):
                break

        return gold_action_seqs



