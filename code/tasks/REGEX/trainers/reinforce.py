import logging
import os
import sys
sys.path.append('..')
import itertools
import json
import re

import torch

from misc import util

from .iliad import IliadTrainer


class ReinforceTrainer(IliadTrainer):

    def do_rollout(self, batch, student, teacher, is_eval, should_log=False):

        src_words = []
        tgt_words = []
        instructions = []
        gold_regexes = []

        batch_size = len(batch)
        for item in batch:
            src_words.append(item['src_word'])
            tgt_words.append(item['tgt_word'])
            instructions.append(item['instruction'])
            gold_regexes.append(item['regex'])

        teacher.receive_simulation_data(batch)

        pred_regexes = student.predict(src_words, instructions, sample=True)

        pred_tgt_words = self.execute_regex(src_words, pred_regexes)

        rewards = teacher.score(pred_tgt_words)

        student.receive(src_words, instructions, pred_regexes, rewards)

        if should_log:
            fmt = '%15s %s'
            logging.info('')
            logging.info(fmt % ('src_word:', src_words[0]))
            logging.info(fmt % ('tgt_word:', tgt_words[0]))
            logging.info(fmt % ('d_star:', ' '.join(instructions[0])))
            logging.info(fmt % ('true regex:', ''.join(gold_regexes[0])))
            logging.info(fmt % ('pred regex:', ''.join(pred_regexes[0])))
            if pred_tgt_words[0] is not None:
                logging.info(fmt % ('pred tgt_word:', pred_tgt_words[0]))
            else:
                logging.info(fmt % ('pred tgt_word:', 'None'))

        stats = {
            'reward'      : self.compute_reward(batch, pred_tgt_words),
            'train_reward': sum(rewards),
            'e_hat_len'   : sum([len(e) for e in pred_regexes]),
            'e_star_len'  : sum([len(e) for e in gold_regexes]),
            'd_star_len'  : sum([len(d) for d in instructions])
        }

        return stats

    def train(self, datasets, student, teacher):

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every
        entropy_weight = self.config.trainer.entropy_weight

        train_info = {
            'i_iter'         : 0,
            'num_examples'   : 0,
            'best_eval_score': -1e9,
            'stats': {
                    'loss'           : 0,
                    'reward'         : 0,
                    'train_reward'   : 0,
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

            loss = student.learn(entropy_weight)
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

                if should_save:

                    # Eval and save best model
                    eval_info = self.evaluate(datasets['val'], student)
                    eval_score = eval_info['score']
                    eval_preds = eval_info['pred']
                    if eval_score > train_info['best_eval_score']:
                        logging.info('New best score: %.1f' % eval_score)
                        train_info['best_eval_score'] = eval_score
                        student.save('best_dev')
                        self.save_preds('best_dev', eval_preds)

                    # Update data indices
                    train_info['data_idx'] = datasets['train'].idx
                    train_info['data_indices'] = datasets['train'].indices

                    # Save last model
                    student.save('last')
                    self.save_train_info('last', train_info)
                    self.save_preds('last', eval_preds)

            if train_info['i_iter'] >= max_iters:
                break








