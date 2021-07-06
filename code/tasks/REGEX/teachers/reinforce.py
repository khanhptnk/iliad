import os
import sys
import editdistance


class ReinforceTeacher(object):

    def __init__(self, config):
        self.config = config
        self.reward_metric = self.config.teacher.reward_metric

    def receive_simulation_data(self, batch):
        self.gold_tgt_words = []
        for item in batch:
            self.gold_tgt_words.append(item['tgt_word'])

    def score(self, pred_tgt_words):

        rewards = []
        for pred_tgt, gold_tgt in zip(pred_tgt_words, self.gold_tgt_words):
            if self.reward_metric == 'score':
                rewards.append(pred_tgt == gold_tgt)
            else:
                assert self.reward_metric == 'editdistance'
                if pred_tgt is None:
                    d = len(gold_tgt)
                else:
                    d = editdistance.eval(pred_tgt, gold_tgt)
                r = (len(gold_tgt) - d) / len(gold_tgt)
                rewards.append(r)

        return rewards




