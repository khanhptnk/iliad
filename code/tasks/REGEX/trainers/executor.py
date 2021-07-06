import logging
import os
import sys
sys.path.append('..')
import itertools
import json
import random

import torch

from misc import util


class ExecutorTrainer(object):

    def __init__(self, config):
        self.config = config
        self.random = random.Random(self.config.seed)

    def do_rollout(self, batch, executor, is_eval):

        src_words = []
        tgt_words = []
        instructions = []

        batch_size = len(batch)
        for item in batch:
            src_words.append(item['src_word'])
            tgt_words.append(item['tgt_word'])
            instructions.append(item['instruction'])

        executor.init(src_words, instructions, is_eval)

        t = 0
        golds = [None] * batch_size
        while not executor.has_terminated():
            for i in range(batch_size):
                if t + 1 < len(tgt_words[i]):
                    golds[i] = tgt_words[i][t + 1]
                else:
                    golds[i] = '<PAD>'
            executor.act(gold_actions=golds)
            t += 1

    def train(self, datasets, executor):

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every

        i_iter = 0
        total_loss = 0
        best_eval_loss = 1e9
        best_eval_acc = -1e9

        for batch in datasets['train'].iterate_batches():

            i_iter += 1

            self.do_rollout(batch, executor, False)

            loss = executor.learn()
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
                executor.save('last')

                # Save best model
                eval_info = self.evaluate(datasets['val'], executor)
                eval_loss = eval_info['loss']
                eval_acc = eval_info['acc']
                eval_preds = eval_info['pred']

                if eval_acc > best_eval_acc:
                    logging.info('New best acc: %.1f' % eval_acc)
                    best_eval_acc = eval_acc
                    executor.save('best_dev')
                    self.save_preds('best_dev', eval_preds)
                self.save_preds('last', eval_preds)

            if i_iter >= max_iters:
                break

    def evaluate(self, dataset, executor, save_pred=False):

        losses = []
        all_preds = []
        is_match = []

        for i, batch in enumerate(dataset.iterate_batches()):

            with torch.no_grad():

                # Compute loss on unseen data
                self.do_rollout(batch, executor, False)
                loss = executor.compute_loss().item()
                losses.append(loss)

                # Make predictions
                src_words = [item['src_word'] for item in batch]
                instructions = [item['instruction'] for item in batch]
                preds = executor.predict(src_words, instructions)

            for item, pred in zip(batch, preds):

                new_item = {}
                new_item.update(item)

                pred = ''.join(pred)
                gold = ''.join(new_item['tgt_word'])
                new_item['pred'] = pred
                is_match.append(gold == pred)
                new_item['is_match'] = is_match[-1]
                new_item['src_word'] = ''.join(new_item['src_word'])
                new_item['tgt_word'] = ''.join(new_item['tgt_word'])
                new_item['instruction'] = ' '.join(new_item['instruction'])

                all_preds.append(new_item)

        avg_loss = sum(losses) / len(losses)
        acc = sum(is_match) / len(is_match) * 100

        log_str = 'Evaluation on %s: ' % dataset.split
        log_str += 'loss = %.1f' % avg_loss
        log_str += ', acc = %.1f' % acc

        logging.info(log_str)

        if save_pred:
            self.save_preds(dataset.split, all_preds)

        eval_info = {
                'acc' : acc,
                'loss': avg_loss,
                'pred': all_preds,
            }

        return eval_info

    def save_preds(self, filename, all_preds):
        file_path = '%s/%s' % (self.config.experiment_dir, filename + '.pred')
        with open(file_path, 'w') as f:
            json.dump(all_preds, f, indent=2)
        logging.info('Saved eval info to %s' % file_path)

