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


class DescriberTrainer(object):

    def __init__(self, config):
        self.config = config

    def do_rollout(self, batch, student, is_eval):

        init_poses = []
        paths = []
        instructions = []

        batch_size = len(batch)

        for item in batch:
            pose = (item['scan'], item['path'][0], item['heading'], 0)
            init_poses.append(pose)
            paths.append(item['path'])
            instructions.append(item['instruction'])

        student.init(init_poses, paths, is_eval)

        t = 0
        teacher_actions = [None] * batch_size
        while not student.has_terminated():
            for i in range(batch_size):
                if t < len(instructions[i]):
                    teacher_actions[i] = instructions[i][t]
                else:
                    teacher_actions[i] = '<EOS>'
            student_actions = student.act(teacher_actions=teacher_actions)
            t += 1

    def train(self, datasets, student, executor, teacher):

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every
        metric_name = self.config.trainer.main_metric_name

        i_iter = 0
        total_loss = 0

        best_metric = {
                'val'  : teacher.init_metric_value(metric_name),
            }

        for batch in itertools.cycle(datasets['train'].iterate_batches()):

            i_iter += 1

            self.do_rollout(batch, student, False)

            loss = student.learn()
            total_loss += loss

            if i_iter % log_every == 0:

                avg_loss = total_loss / log_every
                total_loss = 0

                log_str = 'Train iter %d (%d%%): ' % \
                    (i_iter, i_iter / max_iters * 100)
                log_str += 'loss = %.4f' % avg_loss

                logging.info('')
                logging.info(log_str)

                # Save last model
                student.save('last')

                # Save best model
                for split in best_metric.keys():
                    eval_info = self.evaluate(
                        datasets[split], student, executor, teacher)
                    eval_preds = eval_info['pred']
                    self.save_preds('last_%s' % split, eval_preds)

                    eval_metric = eval_info['metric'][metric_name]
                    if teacher.is_better(metric_name, eval_metric, best_metric[split]):
                        logging.info('New best %s: %.1f' % (split, eval_metric))
                        best_metric[split] = eval_metric
                        save_name = 'best_%s' % split
                        student.save(save_name)
                        self.save_preds(save_name, eval_preds)

            if i_iter >= max_iters:
                break

    def evaluate(self, dataset, student, executor, teacher, save_pred=False):

        all_preds = {}

        with torch.no_grad():
            for i, batch in enumerate(dataset.iterate_batches()):

                # Make predictions
                init_poses = []
                paths = []
                goal_viewpoints = []

                batch_size = len(batch)
                for item in batch:
                    pose = (item['scan'], item['path'][0], item['heading'], 0)
                    init_poses.append(pose)
                    paths.append(item['path'])
                    goal_viewpoints.append(item['path'][-1])

                pred_instructions, pred_paths, metrics = \
                    student.pragmatic_predict(init_poses, paths, executor, teacher)

                zipped_info = zip(
                    batch, pred_instructions, pred_paths, metrics)
                for item, instr, path, metric in zipped_info:
                    new_item = { 'pred_instruction' : ' '.join(instr),
                                 'pred_path' : path
                               }
                    new_item.update(metric)
                    new_item.update(item)
                    if 'new_instructions' in new_item:
                        del new_item['new_instructions']
                    if 'chunk_view' in new_item:
                        del new_item['chunk_view']
                    instr_id = new_item['instr_id']
                    all_preds[instr_id] = new_item

        all_metrics = defaultdict(list)
        for item in all_preds.values():
            for metric_name in teacher.get_metrics():
                all_metrics[metric_name].append(item[metric_name])

        avg_metric = {}
        for metric_name in teacher.get_metrics():
            avg_metric[metric_name] = np.average(all_metrics[metric_name])

        log_str = 'Evaluation on %s: ' % dataset.split
        log_str += teacher.format_metrics(avg_metric)
        logging.info(log_str)

        if save_pred:
            self.save_preds(dataset.split, all_preds)

        eval_info = {
                'metric': avg_metric,
                'pred'  : all_preds,
            }

        return eval_info

    def save_preds(self, filename, all_preds):
        file_path = os.path.join(self.config.experiment_dir, filename + '.pred')
        with open(file_path, 'w') as f:
            json.dump(all_preds, f, indent=2)
        logging.info('Saved eval info to %s' % file_path)

