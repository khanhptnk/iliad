import logging
import os
import sys
sys.path.append('..')
import itertools
import json
import random

import torch

from misc import util


class DescriberTrainer(object):

    def __init__(self, config):
        self.config = config
        self.random = random.Random(self.config.seed)

    def do_rollout(self, batch, describer, is_eval):

        examples = []
        instructions = []

        batch_size = len(batch)
        for item in batch:
            examples.append([])
            for src_word, tgt_word in item['examples']:
                word = src_word + '@' + tgt_word
                examples[-1].append(word)

            instructions.append(item['instruction'])

        describer.init(examples, is_eval)

        t = 0
        golds = [None] * batch_size
        while not describer.has_terminated():
            for i in range(batch_size):
                if t + 1 < len(instructions[i]):
                    golds[i] = instructions[i][t + 1]
                else:
                    golds[i] = '<PAD>'
            describer.act(gold_actions=golds)
            t += 1

    def train(self, datasets, describer, executor):

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every

        i_iter = 0
        total_loss = 0
        best_eval_score = -1e9

        for batch in datasets['train'].iterate_batches():

            i_iter += 1

            self.do_rollout(batch, describer, False)

            loss = describer.learn()
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
                describer.save('last')

                # Save best model
                eval_info = self.evaluate(datasets['val'], describer, executor)
                eval_score = eval_info['score']
                eval_preds = eval_info['pred']
                if eval_score > best_eval_score:
                    logging.info('New best score: %.1f' % eval_score)
                    best_eval_score = eval_score
                    describer.save('best_dev')
                    self.save_preds('best_dev', eval_preds)
                self.save_preds('last', eval_preds)

            if i_iter >= max_iters:
                break

    def evaluate(self, dataset, describer, executor, save_pred=False):

        all_scores = []
        all_preds = []

        for batch in dataset.iterate_batches():

            with torch.no_grad():
                # Make predictions
                examples = []
                for item in batch:
                    examples.append([])
                    for src_word, tgt_word in item['examples']:
                        word = src_word + '@' + tgt_word
                        examples[-1].append(word)
                preds, pred_tgt_words, scores = describer.pragmatic_predict(
                    examples, executor)
                all_scores.extend(scores)

            for item, pred, pred_tgt in zip(batch, preds, pred_tgt_words):
                new_item = { 'pred': ' '.join(pred) }
                new_item.update(item)
                new_item['src_word'] = ''.join(new_item['src_word'])
                new_item['tgt_word'] = ''.join(new_item['tgt_word'])
                new_item['instruction'] = ' '.join(new_item['instruction'])
                new_item['pred_tgt_word'] = ''.join(pred_tgt[0])
                new_item['is_correct'] = \
                    new_item['tgt_word'] == new_item['pred_tgt_word']
                del new_item['examples']
                all_preds.append(new_item)

        n_examples = len(batch[0]['examples'])
        score = sum(all_scores) / (len(all_scores) * n_examples) * 100

        log_str = 'Evaluation on %s: ' % dataset.split
        log_str += 'score = %.1f' % score
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

