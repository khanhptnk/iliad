import logging
import os
import sys
sys.path.append('..')
import itertools
import json
import re
from termcolor import colored

import torch

from misc import util


class IliadTrainer(object):

    def __init__(self, config):
        self.config = config

    def execute_regex(self, src_words, regexes):

        tgt_words = []
        for i, regex in enumerate(regexes):
            regex = ''.join(regex)[1:-1]
            if '@' in regex and len(regex.split('@')) == 2:
                before, after = regex.split('@')
                if 'C' in after or 'V' in after or '(' in after or ')' in after:
                    tgt_words.append(None)
                    continue
                before = before.replace('C', '[^aeiou]').replace('V', '[aeiou]')
                after = '\\1' + after + '\\3'
                src_word = ''.join(src_words[i])[1:-1]
                try:
                    tgt_word = '<' + re.sub(before, after, src_word) + '>'
                    tgt_words.append(tgt_word)
                except:
                    tgt_words.append(None)
            else:
                tgt_words.append(None)

        assert len(tgt_words) == len(src_words)

        return tgt_words

    def do_rollout(self, batch, student, teacher, is_eval, should_log=False):

        batch_size = len(batch)
        n_examples = len(batch[0]['examples'])

        src_words = []
        example_src_words = []
        example_tgt_words = []
        instructions = []

        for item in batch:

            for src_word, tgt_word in item['examples']:
                example_src_words.append(src_word)
                example_tgt_words.append(tgt_word)

            src_words.append(item['src_word'])
            instructions.append(item['instruction'])

        teacher.receive_simulation_data(batch)
        student.reset()

        if should_log:
            fmt = '%15s %s'
            logging.info('')
            logging.info(fmt % ('src_word:', example_src_words[0]))
            logging.info(fmt % ('tgt_word:', example_tgt_words[0]))
            logging.info(fmt % ('true regex:', ''.join(batch[0]['regex'])))
            logging.info(fmt % ('d_star:', ' '.join(instructions[0])))

        # Decode the first time (with exploration) for language learning
        pred_regexes = student.predict(
        src_words, instructions, 'exploration', sample=True)

        if should_log:
            logging.info(fmt % ('pred regex:', ''.join(pred_regexes[0])))

        pred_example_regexes = []
        for regex in pred_regexes:
            pred_example_regexes.extend([regex] * n_examples)

        pred_example_tgt_words = self.execute_regex(
            example_src_words, pred_example_regexes)

        queried_examples = []
        queried_indices = []
        for i in range(batch_size):

            start_ix = i * n_examples
            end_ix = (i + 1) * n_examples

            all_equal = True
            tgt_none = False
            for j in range(start_ix, end_ix):
                if pred_example_tgt_words[j] is None:
                    tgt_none = True
                else:
                    src_word = example_src_words[j]
                    tgt_word = ''.join(pred_example_tgt_words[j])
                    if src_word != tgt_word:
                        all_equal = False

            # Don't query when no changes occur
            # Don't query when regex is invalid
            if not all_equal and not tgt_none:
                queried_indices.append(i)
                queried_examples.append([])
                for j in range(start_ix, end_ix):
                    src_word = example_src_words[j]
                    tgt_word = ''.join(pred_example_tgt_words[j])
                    queried_examples[-1].append(src_word + '@' + tgt_word)

        if should_log:
            if pred_example_tgt_words[0] is not None:
                logging.info(fmt % ('pred tgt word:', ''.join(pred_example_tgt_words[0])))
            else:
                logging.info(fmt % ('pred tgt word:', 'None'))

        descriptions, num_gt_d_hat = teacher.describe(
            queried_examples, queried_indices)

        if should_log:
            if 0 in queried_indices and descriptions[0] is not None:
                logging.info(fmt % ('d_hat:', ' '.join(descriptions[0])))
            else:
                logging.info(fmt % ('d_hat:', 'None'))

        added_src_words = []
        added_gold_instructions = []
        added_instructions = []
        added_regexes = []

        for i, ix in enumerate(queried_indices):
            if descriptions[i] is not None:
                start_ix = ix * n_examples
                end_ix = (ix + 1) * n_examples
                for j in range(start_ix, end_ix):
                    added_src_words.append(example_src_words[j])
                    added_gold_instructions.append(instructions[ix])
                    added_instructions.append(descriptions[i])
                    added_regexes.append(pred_regexes[ix])

        student.receive(added_src_words, added_gold_instructions, added_instructions, added_regexes)

        pred_tgt_words = self.execute_regex(src_words, pred_regexes)

        stats = {
            'reward'      : self.compute_reward(batch, pred_tgt_words),
            'e_hat_len'   : sum([len(e) for e in pred_regexes]),
            'd_star_len'  : sum([len(d) for d in instructions]),
            'd_hat_len'   : sum([len(d) for d in descriptions if d is not None]),
            'num_d_hat'   : sum([d is not None for d in descriptions]),
            'num_gt_d_hat': num_gt_d_hat
        }

        return stats

    def compute_reward(self, batch, pred_tgt_words):

        total_reward = 0
        for item, pred_tgt_word in zip(batch, pred_tgt_words):
            gold_tgt_word = item['tgt_word']
            total_reward += pred_tgt_word == gold_tgt_word

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
        unsup_weight_config = self.config.trainer.unsup_weight

        train_info = {
            'i_iter'         : 0,
            'num_examples'   : 0,
            'best_eval_score': -1e9,
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

        if self.config.resume and \
            hasattr(self.config.student.model, 'load_from') and \
            'unsupervised' not in self.config.student.model.load_from:

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

    def evaluate(self, dataset, student, save_pred=False):

        all_scores = []
        all_preds = []

        stats = {
                'num_examples': 0,
                'reward': 0,
                'e_hat_len': 0,
                'd_star_len': 0,
            }

        for i, batch in enumerate(dataset.iterate_batches()):

            with torch.no_grad():

                # Make predictions
                src_words = [item['src_word'] for item in batch]
                instructions = [item['instruction'] for item in batch]
                preds = student.predict(src_words, instructions)
                pred_tgt_words = [None] * len(batch)

                scores = [None] * len(batch)
                for i, regex in enumerate(preds):
                    src_word = ''.join(batch[i]['src_word'])[1:-1]
                    tgt_word = ''.join(batch[i]['tgt_word'])[1:-1]
                    regex = ''.join(regex)[1:-1]
                    if '@' not in regex or len(regex.split('@')) != 2:
                        scores[i] = 0
                    else:
                        before, after = regex.split('@')
                        before = before.replace('C', '[^aeiou]').replace('V', '[aeiou]')
                        after = '\\1' + after + '\\3'
                        try:
                            pred_tgt_word = re.sub(before, after, src_word)
                            scores[i] = pred_tgt_word == tgt_word
                            pred_tgt_words[i] = pred_tgt_word
                        except:
                            scores[i] = 0
                all_scores.extend(scores)

                stats['num_examples'] += len(batch)
                stats['reward'] += sum(scores)
                stats['e_hat_len'] += sum([len(e) for e in preds])
                stats['d_star_len'] += sum([len(d) for d in instructions])

            zipped_info = zip(batch, preds, scores, pred_tgt_words)
            for item, pred, score, pred_tgt_word in zipped_info:
                new_item = { 'pred': ''.join(pred)  }
                new_item.update(item)
                new_item['score'] = score
                new_item['pred_tgt_word'] = pred_tgt_word
                new_item['src_word'] = ''.join(new_item['src_word'])
                new_item['tgt_word'] = ''.join(new_item['tgt_word'])
                new_item['instruction'] = ' '.join(new_item['instruction'])
                new_item['regex'] = ''.join(new_item['regex'])
                all_preds.append(new_item)

        score = sum(all_scores) / len(all_scores) * 100

        log_str = 'Evaluation on %s: ' % dataset.split
        log_str += 'score = %.1f' % score
        log_str += ', reward = %.4f'     % (stats['reward'] / stats['num_examples'])
        log_str += ', e_hat_len = %.4f'  % (stats['e_hat_len'] / stats['num_examples'])
        log_str += ', d_star_len = %.4f' % (stats['d_star_len'] / stats['num_examples'])
        log_str += ', num_examples = %d' % stats['num_examples']
        logging.info('')
        logging.info(log_str)

        if save_pred:
            self.save_preds(dataset.split, all_preds)

        eval_info = {
                'score': score,
                'pred': all_preds,
            }

        return eval_info

    def save_preds(self, filename, all_preds):
        file_path = os.path.join(self.config.experiment_dir, filename + '.pred')
        with open(file_path, 'w') as f:
            json.dump(all_preds, f, indent=2)
        logging.info('Saved eval info to %s' % file_path)

