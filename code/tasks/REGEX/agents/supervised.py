import os
import sys
import json
import logging

import torch
import torch.nn as nn
import torch.distributions as D

import models


class SupervisedStudent(object):

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.task_vocab = config.word_vocab
        self.src_vocab = config.char_vocab
        self.tgt_vocab = config.regex_vocab

        model_config = config.student.model
        model_config.device = config.device
        model_config.task_vocab_size = len(self.task_vocab)
        model_config.src_vocab_size = len(self.src_vocab)
        model_config.tgt_vocab_size = len(self.tgt_vocab)

        model_config.task_pad_idx = self.task_vocab['<PAD>']
        model_config.src_pad_idx = self.src_vocab['<PAD>']
        model_config.tgt_pad_idx = self.tgt_vocab['<PAD>']
        model_config.n_actions = len(self.tgt_vocab)

        self.model = models.load(model_config).to(self.device)

        logging.info('model: ' + str(self.model))

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=model_config.learning_rate)

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.tgt_pad_idx)

    def _to_tensor(self, x):
        return torch.tensor(x).to(self.device)

    def _index_and_pad(self, xs, vocab):

        # Init instructed model
        encodings = []
        masks = []
        for x in xs:
            encodings.append([vocab[w] for w in x])
            masks.append([0] * len(encodings[-1]))

        # Padding
        max_len = max([len(encoding) for encoding in encodings])

        for i, encoding in enumerate(encodings):
            encoding.extend([vocab['<PAD>']] * (max_len - len(encoding)))

        for mask in masks:
            mask.extend([1] * (max_len - len(mask)))

        encodings = self._to_tensor(encodings).long()
        masks = self._to_tensor(masks).bool()

        return encodings, masks

    def init(self, srcs, instructions, is_eval):

        if is_eval:
            self.model.eval()
        else:
            self.model.train()

        self.is_eval = is_eval
        self.batch_size = len(srcs)

        self.pred_action_seqs = [['<'] for _ in range(self.batch_size)]
        self.gold_action_seqs = []
        self.action_logit_seqs = []
        self.terminated = [False] * self.batch_size

        task_encodings, task_masks = self._index_and_pad(instructions, self.task_vocab)
        src_encodings, src_masks = self._index_and_pad(srcs, self.src_vocab)

        self.model.encode(task_encodings, task_masks, src_encodings, src_masks)

        self.prev_actions = [self.tgt_vocab['<']] * self.batch_size
        self.prev_actions = self._to_tensor(self.prev_actions)

        self.timer = self.config.student.max_timesteps

    def act(self, gold_actions=None, sample=False, debug=False):

        action_logits = self.model.decode(self.prev_actions)
        self.action_logit_seqs.append(action_logits)

        if self.is_eval:
            if sample:
                pred_actions = D.Categorical(logits=action_logits).sample()
                d = action_logits.softmax(dim=1)
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

    def predict(self, src_words, instructions, sample=False):

        self.init(src_words, instructions, True)
        while not self.has_terminated():
            self.act(sample=sample)

        return self.get_action_seqs()

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
        #self.optim.load_state_dict(ckpt['optim_state_dict'])
        logging.info('Loaded model from %s' % file_path)





