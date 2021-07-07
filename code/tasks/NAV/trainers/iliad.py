import logging
import os
import sys
import itertools
import json

import torch

from misc import util
from .executor import ExecutorTrainer


class IliadTrainer(ExecutorTrainer):

    def _remove_loops_in_paths(self, paths):
        new_paths = []
        for path in paths:
            new_path = []
            visited_viewpoints = set()
            for u in path:
                if u in visited_viewpoints:
                    while new_path:
                        v = new_path.pop()
                        visited_viewpoints.remove(v)
                        if v == u:
                            break
                visited_viewpoints.add(u)
                new_path.append(u)
            new_paths.append(new_path)
        return new_paths

    def do_rollout(self, batch, student, teacher, is_eval, should_log=False):

        batch_size = len(batch)

        init_poses = []
        goal_viewpoints = []
        instructions = []

        batch_size = len(batch)

        for item in batch:
            pose = (item['scan'], item['path'][0], item['heading'], 0)
            init_poses.append(pose)
            goal_viewpoints.append(item['path'][-1])
            instructions.append(item['instruction'])

        if should_log:
            logging.info('')
            logging.info('    instr_id %s   scan %s   heading %f' %
                (batch[0]['instr_id'], batch[0]['scan'], batch[0]['heading']))
            logging.info('    gold path: %s' % str(batch[0]['path']))

        teacher.receive_simulation_data(batch)
        student.reset()

        pred_paths, _ = student.predict(
            init_poses, instructions, sample=True, model_name='exploration')

        pred_paths = self._remove_loops_in_paths(pred_paths)

        descriptions, num_gt_d_hat = teacher.describe(init_poses, pred_paths)
        is_valid = [d != ['<PAD>'] for d in descriptions]

        if should_log:
            logging.info('    pred path: %s' % str(pred_paths[0]))
            logging.info('    instruction: %s' % ' '.join(instructions[0]))
            logging.info('    description: %s' % ' '.join(descriptions[0]))
            metric = teacher.eval(batch[0]['scan'], batch[0]['path'], pred_paths[0])
            logging.info('    metric: %s' % str(metric))

        student.receive(init_poses, instructions, descriptions, pred_paths, is_valid)

        stats = {
            'reward'      : self.compute_reward(teacher, batch, pred_paths),
            'e_hat_len'   : sum([len(e) for e in pred_paths]),
            'd_star_len'  : sum([len(d) for d in instructions]),
            'd_hat_len'   : sum([len(d) for d in descriptions if d != ['<PAD>']]),
            'num_d_hat'   : sum([d != ['<PAD>'] for d in descriptions]),
            'num_gt_d_hat': num_gt_d_hat
        }

        return stats

    def compute_reward(self, teacher, batch, pred_paths):

        total_reward = 0
        for i, pred_path in enumerate(pred_paths):
            item = batch[i]
            scan = item['scan']
            gold_path = item['path']
            metric = teacher.eval(scan, pred_path, gold_path)
            total_reward += metric['score']

        return total_reward

    def save_train_info(self, name, train_info):
        file_path = '%s/%s' % (self.config.experiment_dir, name + '.info')
        torch.save(train_info, file_path)
        logging.info('Saved train info to %s' % file_path)

    def load_train_info(self, file_path):
        file_path = file_path.replace('ckpt', 'info')
        train_info = torch.load(file_path)
        return train_info

    def train(self, datasets, student, teacher):

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every
        log_rate  = self.config.trainer.log_rate
        metric_name = self.config.trainer.main_metric_name
        unsup_weight_config = self.config.trainer.unsup_weight

        train_info = {
            'i_iter'         : 0,
            'num_examples'   : 0,
            'best_eval_score': teacher.init_metric_value(metric_name),
            'unsup_weight'   : unsup_weight_config.init,
            'stats': {
                    'loss'           : 0,
                    'reward'         : 0,
                    'e_hat_len'      : 0,
                    'd_star_len'     : 0,
                    'd_hat_len'      : 0,
                    'num_d_hat'      : 0,
                    'num_gt_d_hat'   : 0,
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
            should_save = (train_info['i_iter'] % (log_every * log_rate) == 0)

            stats = self.do_rollout(batch, student, teacher, False,
                should_log=should_log)

            for k in stats:
                train_info['stats'][k] += stats[k]

            loss = student.learn(train_info['unsup_weight'])
            train_info['stats']['loss'] += loss

            # Decay unsup weight
            if train_info['i_iter'] % unsup_weight_config.decay_every == 0:
                train_info['unsup_weight'] = max(
                    train_info['unsup_weight'] * unsup_weight_config.rate,
                    unsup_weight_config.min)
                logging.info('')
                logging.info('Train iter %d: decay unsup weight = %.5f' %
                    (train_info['i_iter'], train_info['unsup_weight']))

            if should_log:

                log_str = 'Train iter %d (%d%%): ' % (
                     train_info['i_iter'],
                     train_info['i_iter'] / max_iters * 100)

                stat_strs = []
                stat_strs.append('lambda = %.5f' % train_info['unsup_weight'])
                stat_strs.append('num_examples = %d' % train_info['num_examples'])
                for stat_name, stat_value in train_info['stats'].items():
                    if stat_name == 'loss':
                        stat = stat_value / train_info['i_iter']
                    elif stat_name == 'd_hat_len':
                        stat = stat_value / train_info['stats']['num_d_hat']
                    elif stat_name in ['reward', 'num_d_hat', 'num_gt_d_hat']:
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
