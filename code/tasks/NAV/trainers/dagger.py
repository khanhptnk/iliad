import logging
import os
import sys
import itertools
import json
import re
import numpy as np
from collections import defaultdict

import torch

from misc import util

from .iliad import IliadTrainer


class DaggerTrainer(IliadTrainer):

    def do_rollout(self, batch, student, teacher, is_eval, should_log=False):

        init_poses = []
        goal_viewpoints = []
        instructions = []
        gold_paths = []

        batch_size = len(batch)

        for item in batch:
            pose = (item['scan'], item['path'][0], item['heading'], 0)
            init_poses.append(pose)
            goal_viewpoints.append(item['path'][-1])
            instructions.append(item['instruction'])
            gold_paths.append(item['path'])

        teacher.receive_simulation_data(batch)

        pred_paths, _ = student.predict(init_poses, instructions)

        teacher_action_seqs = teacher.demonstrate(init_poses, instructions, pred_paths)

        student.receive(init_poses, instructions, pred_paths, teacher_action_seqs)

        if should_log:
            logging.info('    instr_id %s   scan %s   heading %f' %
                (batch[0]['instr_id'], batch[0]['scan'], batch[0]['heading']))
            logging.info('    instruction: %s' % ' '.join(instructions[0]))
            logging.info('    pred path: %s' % str(pred_paths[0]))
            logging.info('    true path: %s' % str(batch[0]['path']))
            metric = teacher.eval(batch[0]['scan'], batch[0]['path'], pred_paths[0])
            logging.info('    metric: %s' % str(metric))

        stats = {
            'reward'    : self.compute_reward(teacher, batch, pred_paths),
            'e_hat_len' : sum([len(e) for e in pred_paths]),
            'e_star_len': sum([len(e) for e in gold_paths]),
            'd_star_len': sum([len(d) for d in instructions])
        }

        return stats

    def train(self, datasets, student, teacher):

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every
        metric_name = self.config.trainer.main_metric_name

        train_info = {
            'i_iter'         : 0,
            'num_examples'   : 0,
            'best_eval_score': teacher.init_metric_value(metric_name),
            'stats': {
                    'loss'           : 0,
                    'reward'         : 0,
                    'e_hat_len'      : 0,
                    'e_star_len'     : 0,
                    'd_star_len'     : 0,
                }
        }

        data_iter = datasets['train'].iterate_batches()

        if self.config.resume and hasattr(self.config.student.model, 'load_from'):

            train_info = self.load_train_info(
                self.config.student.model.load_from)

            data_iter = datasets['train'].iterate_batches(
                data_idx=train_info['data_idx'],
                data_indices=train_info['data_indices'])

            train_info['data_indices'] = train_info['data_indices'][:10]
            logging.info('Loaded train info %s' % str(train_info))

        for batch in data_iter:

            train_info['i_iter'] += 1
            train_info['num_examples'] += len(batch)

            should_log  = (train_info['i_iter'] % log_every == 0)
            should_save = (train_info['i_iter'] % (log_every * 10) == 0)

            stats = self.do_rollout(batch, student, teacher, False,
                should_log=should_log)

            for k in stats:
                train_info['stats'][k] += stats[k]

            loss = student.learn()
            train_info['stats']['loss'] += loss

            if should_log:

                log_str = 'Train iter %d (%d%%): ' % (
                     train_info['i_iter'],
                     train_info['i_iter'] / max_iters * 100)

                stat_strs = []
                stat_strs.append('num_examples = %d' % train_info['num_examples'])
                for stat_name, stat_value in train_info['stats'].items():
                    if stat_name == 'loss':
                        stat = stat_value / train_info['i_iter']
                    elif stat_name == 'reward':
                        stat = stat_value
                    else:
                        stat = stat_value / train_info['num_examples']
                    stat_strs.append('%s = %.3f' % (stat_name, stat))

                log_str += ', '.join(stat_strs)
                logging.info('')
                logging.info(log_str)

                # Save best model
                if should_save:

                    eval_info = self.evaluate(datasets['val'], student, teacher)
                    eval_preds = eval_info['pred']
                    eval_score = eval_info['metric'][metric_name]

                    if teacher.is_better(metric_name, eval_score, train_info['best_eval_score']):
                        logging.info('New best score: %.1f' % eval_score)
                        train_info['best_eval_score'] = eval_score
                        student.save('best_val')
                        self.save_preds('best_val', eval_preds)

                    # Update data indices
                    train_info['data_idx'] = datasets['train'].idx
                    train_info['data_indices'] = datasets['train'].indices

                    # Save last model
                    student.save('last')
                    self.save_train_info('last', train_info)
                    self.save_preds('last', eval_preds)

            if train_info['i_iter'] >= max_iters:
                break
