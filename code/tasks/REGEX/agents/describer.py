import os
import sys
import json
import logging

import torch
import torch.nn as nn
import torch.distributions as D

import editdistance

import models


class Describer(object):

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.src_vocab = config.char_vocab
        self.tgt_vocab = config.word_vocab

        model_config = config.describer.model
        model_config.device = config.device
        model_config.src_vocab_size = len(self.src_vocab)
        model_config.tgt_vocab_size = len(self.tgt_vocab)

        model_config.src_pad_idx = self.src_vocab['<PAD>']
        model_config.tgt_pad_idx = self.tgt_vocab['<PAD>']
        model_config.n_actions = len(self.tgt_vocab)

        model_config.enc_hidden_size = model_config.hidden_size
        model_config.dec_hidden_size = model_config.hidden_size

        self.model = models.load(model_config).to(self.device)

        logging.info('model: ' + str(self.model))

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=model_config.learning_rate)

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.tgt_pad_idx)
        self.n_samples = self.config.describer.n_samples

    def _to_tensor(self, x):
        return torch.tensor(x).to(self.device)

    def _pad(self, seq, pad_token, max_len):
        return seq.extend([pad_token] * (max_len - len(seq)))

    def _index_and_pad(self, examples, vocab):

        encodings = []
        max_len = 0
        for x in examples:
            encodings.append([])
            for seq in x:
                indices = [vocab[w] for w in seq]
                encodings[-1].append(indices)
                max_len = max(max_len, len(indices))

        for x in encodings:
            for seq in x:
                self._pad(seq, vocab['<PAD>'], max_len)

        encodings = self._to_tensor(encodings).long()

        return encodings

    def init(self, examples, is_eval):

        if is_eval:
            self.model.eval()
        else:
            self.model.train()

        self.is_eval = is_eval
        self.batch_size = len(examples)

        self.pred_action_seqs = [['<'] for _ in range(self.batch_size)]
        self.gold_action_seqs = []
        self.action_logit_seqs = []
        self.terminated = [False] * self.batch_size

        example_encodings = self._index_and_pad(examples, self.src_vocab)
        self.model.encode(example_encodings)

        self.prev_actions = [self.tgt_vocab['<']] * self.batch_size
        self.prev_actions = self._to_tensor(self.prev_actions)

        self.timer = self.config.describer.max_timesteps

    def act(self, gold_actions=None, sample=False):

        action_logits = self.model.decode(self.prev_actions)
        self.action_logit_seqs.append(action_logits)

        if self.is_eval:
            if sample:
                pred_actions = D.Categorical(logits=action_logits).sample()
            else:
                pred_actions = action_logits.max(dim=1)[1]
            self.prev_actions = pred_actions

            pred_actions = pred_actions.tolist()
            for i in range(self.batch_size):
                if not self.terminated[i]:
                    w = self.tgt_vocab.get(pred_actions[i])
                    self.pred_action_seqs[i].append(w)
        else:
            gold_actions = [self.tgt_vocab[w] for w in gold_actions]
            gold_actions = self._to_tensor(gold_actions).long()
            self.gold_action_seqs.append(gold_actions)
            self.prev_actions = gold_actions

        self.timer -= 1

        for i in range(self.batch_size):
            self.terminated[i] |= self.timer <= 0
            self.terminated[i] |= self.prev_actions[i].item() == self.tgt_vocab['>']

    def has_terminated(self):
        return all(self.terminated)

    def get_action_seqs(self):
        return self.pred_action_seqs

    def predict(self, examples):

        self.init(examples, True)

        while not self.has_terminated():
            self.act()

        return self.get_action_seqs()


    def pragmatic_predict(self, examples, executor, n_samples=None):

        def infer_tgt_words(src_words, tgt_words, instructions):
            scores = [0] * batch_size
            pred_tgt_words = [[None] * n_examples for _ in range(batch_size)]
            for j in range(n_examples):
                batch_src_words = [item[j] for item in src_words]
                batch_tgt_words = executor.predict(batch_src_words, instructions)
                for i in range(batch_size):
                    gold = tgt_words[i][j]
                    pred = ''.join(batch_tgt_words[i])
                    scores[i] += gold == pred
                    pred_tgt_words[i][j] = batch_tgt_words[i]
            return scores, pred_tgt_words

        if n_samples is None:
            n_samples = self.n_samples

        batch_size = len(examples)
        n_examples = len(examples[0])

        src_words = []
        tgt_words = []
        for item in examples:
            src_words.append([])
            tgt_words.append([])
            for word in item:
                src_word, tgt_word = word.split('@')
                src_words[-1].append(src_word)
                tgt_words[-1].append(tgt_word)

        best_pred_instructions = self.predict(examples)
        best_scores, best_pred_tgt_words = infer_tgt_words(
            src_words, tgt_words, best_pred_instructions)

        for k in range(n_samples):

            self.init(examples, True)
            while not self.has_terminated():
                self.act(sample=True)
            pred_instructions = self.get_action_seqs()

            curr_scores, curr_pred_tgt_words = infer_tgt_words(
                src_words, tgt_words, pred_instructions)

            for i in range(batch_size):
                if curr_scores[i] > best_scores[i]:
                    best_scores[i] = curr_scores[i]
                    best_pred_instructions[i] = pred_instructions[i]
                    best_pred_tgt_words[i] = curr_pred_tgt_words[i]

        return best_pred_instructions, best_pred_tgt_words, best_scores

    def compute_loss(self):
        loss = 0
        zipped_info = zip(self.action_logit_seqs, self.gold_action_seqs)
        for logits, golds in zipped_info:
            loss += self.loss_fn(logits, golds)
        return loss

    def learn(self):

        assert len(self.gold_action_seqs) == len(self.action_logit_seqs)

        loss = self.compute_loss()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item() / len(self.gold_action_seqs)

    def save(self, name, trajectories=None):
        file_path = os.path.join(self.config.experiment_dir, name + '.ckpt')
        ckpt = { 'model_state_dict': self.model.state_dict(),
                 'optim_state_dict': self.optim.state_dict() }
        torch.save(ckpt, file_path)
        logging.info('Saved %s model to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        logging.info('Loaded model from %s' % file_path)





