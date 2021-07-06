import os
import sys
import json
import logging

import torch
import torch.nn as nn
import torch.distributions as D

import models


class Describer(object):

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.src_vocab = config.char_vocab
        self.tgt_vocab = config.word_vocab

        model_config = config.teacher.describer
        model_config.device = config.device
        model_config.src_vocab_size = len(config.tgt_vocab)
        model_config.tgt_vocab_size = len(config.src_vocab)

        model_config.src_pad_idx = config.src_vocab['<PAD>']
        model_config.tgt_pad_idx = config.tgt_vocab['<PAD>']
        model_config.n_actions = len(config.tgt_vocab)

        model_config.enc_hidden_size = config.teacher.describer.hidden_size
        model_config.dec_hidden_size = config.teacher.describer.hidden_size

        self.model = models.load(model_config).to(self.device)

        logging.info('model: ' + str(self.model))

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=model_config.learning_rate)

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model_config.tgt_pad_idx)

    def _to_tensor(self, x):
        return torch.tensor(x).to(self.device)

    def _index_and_pad(self, x, vocab):

        # Init instructed model
        encodings = []
        masks = []
        for x in xs:
            encodings.append([vocab[w] for w in x]])
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

    def init(self, srcs, is_eval):

        if is_eval:
            self.model.eval()
        else:
            self.model.train()

        self.is_eval = is_eval
        self.batch_size = len(srcs)

        self.pred_action_seqs = [[] for _ in range(self.batch_size)]
        self.gold_action_seqs = []
        self.action_logit_seqs = []
        self.has_terminated = [False] * self.batch_size

        src_encodings, src_masks = self._index_and_pad(srcs, self.src_vocab)
        self.model.encode(src_encodings, src_masks)

        self.prev_actions = [self.src_vocab['<']] * self.batch_size
        self.prev_actions = self._to_tensor(self.prev_actions)

    def act(self, gold_actions):

        action_logits = self.model.decode(self.prev_actions)
        self.action_logit_seqs.append(action_logits)
        pred_actions = action_logits.max(dim=1)[1]

        gold_actions = [self.tgt_vocab[w] for w in gold_actions]
        gold_actions = self._to_tensor(gold_actions).long()
        self.gold_action_seqs.append(gold_actions)

        if self.is_eval:
            self.prev_actions = pred_actions
        else:
            self.prev_actions = gold_actions

        pred_actions = pred_actions.tolist()

        for i in range(self.batch_size):
            if not self.has_terminated[i]:
                w = self.tgt_vocab.get(pred_actions[i])
                self.pred_action_seqs[i].append(w)

        return pred_actions

    def has_terminted(self, i):
        self.has_terminated[i] |= self.prev_actions[i] == self.src_vocab['>']
        return self.has_terminated[i]

    def compute_loss(self):
        loss = 0
        zipped_info = zip(self.action_logit_seqs, self.gold_action_seqs)
        for logits, golds in zipped_info:
            loss += self.loss_fn(logits, golds)
        return loss

    def learn(self):

        assert len(self.ref_action_seqs) == len(self.action_logit_seqs)

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





