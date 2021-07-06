import os
import sys
sys.path.append('..')
import random
from termcolor import colored

import agents


class IliadTeacher(object):

    def __init__(self, config):

        self.describer = agents.load(config, 'describer')
        self.executor  = agents.load(config, 'executor')
        self.random = random.Random(config.seed)

    def receive_simulation_data(self, batch):
        self.equiv_tasks = [item['equiv_instructions'] for item in batch]
        self.gold_tgt_word_sets = []
        for item in batch:
            self.gold_tgt_word_sets.append([])
            for _, tgt_word in item['examples']:
                self.gold_tgt_word_sets[-1].append(tgt_word)

    def describe(self, example_sets, example_indices):

        if not example_sets:
            return [], 0

        pred_tasks, pred_tgt_words, scores = self.describer.pragmatic_predict(
            example_sets, self.executor)

        n_examples = len(example_sets[0])

        for i, score in enumerate(scores):
            if score < n_examples:
                pred_tasks[i] = None

        assert len(pred_tasks) == len(example_sets)

        descriptions = [None] * len(pred_tasks)
        zipped_info = zip(example_sets, example_indices)

        num_gt_d_hat = 0

        for i, (examples, idx) in enumerate(zipped_info):

            gold_tgt_words = self.gold_tgt_word_sets[idx]
            equiv_tasks = self.equiv_tasks[idx]

            correct = 0
            for pair, gold_tgt_word in zip(examples, gold_tgt_words):
                src_word, pred_tgt_word = pair.split('@')
                correct += pred_tgt_word == gold_tgt_word

            if correct == n_examples:
                num_gt_d_hat += 1
                d = self.random.choice(equiv_tasks)
                descriptions[i] = d
            else:
                descriptions[i] = pred_tasks[i]

        return descriptions, num_gt_d_hat



