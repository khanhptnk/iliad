import os
import sys
import json
import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import models
import worlds

from .executor import Executor


class IliadStudent(Executor):

    STOP = 0

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.vocab = config.vocab

        self.world = worlds.load(config)

        model_config = config.student.model
        model_config.device = config.device
        model_config.vocab_size = len(self.vocab)
        model_config.loc_embed_size = config.world.loc_embed_size
        model_config.max_instruction_length = config.student.max_instruction_length

        model_config.pad_idx = self.vocab['<PAD>']

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.max_instruction_length = config.student.max_instruction_length

        self.exploration_model = models.load(model_config).to(self.device)
        self.execution_model   = models.load(model_config).to(self.device)

        logging.info('exploration model: ' + str(self.exploration_model))
        logging.info('execution model: '   + str(self.execution_model))

        self.optim = torch.optim.Adam(
                list(self.exploration_model.parameters()) +
                list(self.execution_model.parameters()),
                lr=model_config.learning_rate
            )

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

    def _to_tensor(self, x):
        return torch.tensor(x).to(self.device)

    def _to_tensor_from_numpy(self, x):
        return torch.from_numpy(x).to(self.device)

    def _index_and_pad(self, xs, vocab, reverse=True):

        # Init instructed model
        encodings = []
        masks = []
        for x in xs:
            x = x[:self.max_instruction_length] + ['<EOS>']
            encodings.append([vocab[w] for w in x])
            if reverse:
                encodings[-1] = list(reversed(encodings[-1]))
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

    def _nav_action_variable(self, states):
        max_num_a = max(len(state.adj_loc_list) for state in states)
        invalid = np.zeros((self.batch_size, max_num_a), np.uint8)
        action_embed_size = states[0].action_embeddings.shape[-1]
        action_embeds = np.zeros(
            (self.batch_size, max_num_a, action_embed_size), dtype=np.float32)
        for i, state in enumerate(states):
            num_a = len(state.adj_loc_list)
            invalid[i, num_a:] = 1
            action_embeds[i, :num_a, :] = state.action_embeddings

        action_embeds = self._to_tensor_from_numpy(action_embeds).float()
        invalid = self._to_tensor_from_numpy(invalid).bool()

        return action_embeds, invalid

    def set_model(self, name):
        if name == 'exploration':
            self.model = self.exploration_model
        else:
            assert name == 'execution', name
            self.model = self.execution_model

    def reset(self):
        self.train_init_poses = []
        self.train_gold_instructions = []
        self.train_instructions = []
        self.train_paths = []
        self.train_is_valid = []

    def init(self, model_name, init_poses, instructions, is_eval):
        self.set_model(model_name)
        return super(IliadStudent, self).init(init_poses, instructions, is_eval)

    def act(self, states, teacher_actions=None, sample=False):

        curr_view_features = [state.curr_view_features for state in states]
        curr_view_features = self._to_tensor_from_numpy(
            np.stack(curr_view_features))

        all_action_embeds, logit_masks = self._nav_action_variable(states)

        self.text_dec_h, self.state_dec_h, self.dec_time, action_logits = \
            self.model.decode(
                self.text_dec_h,
                self.state_dec_h,
                self.dec_time,
                self.prev_action_embeds,
                all_action_embeds,
                self.instructions,
                self.instruction_masks,
                curr_view_features,
                logit_masks
            )

        self.action_logit_seqs.append(action_logits)

        if self.is_eval:
            if sample:
                pred_actions = D.Categorical(logits=action_logits).sample().tolist()
            else:
                pred_actions = action_logits.max(dim=1)[1].tolist()
            self.prev_actions = pred_actions
            for i in range(self.batch_size):
                if not self.terminated[i]:
                    self.pred_action_seqs[i].append(pred_actions[i])
        else:
            pred_actions = teacher_actions
            self.prev_actions = pred_actions
            teacher_actions = self._to_tensor(teacher_actions).long()
            for i in range(self.batch_size):
                if self.terminated[i]:
                    teacher_actions[i] = -1
            self.teacher_action_seqs.append(teacher_actions)

        self.timer -= 1

        for i in range(self.batch_size):
            self.terminated[i] |= self.timer <= 0
            self.terminated[i] |= self.prev_actions[i] == self.STOP

        self.prev_action_embeds = all_action_embeds[np.arange(self.batch_size), pred_actions, :].detach()

        return self.prev_actions

    def predict(self, init_poses, instructions, sample=False,
            model_name='execution'):

        with torch.no_grad():
            states = self.init(model_name, init_poses, instructions, True)
            paths = [[state.viewpoint] for state in states]
            poses = [[pose] for pose in init_poses]
            while not self.has_terminated():
                pred_actions = self.act(states, sample=sample)
                states = states.step(pred_actions)
                for i, state in enumerate(states):
                    pose = (state.scan, state.viewpoint, state.heading, state.elevation)
                    if not self.terminated[i]:
                        poses[i].append(pose)
                        if state.viewpoint != paths[i][-1]:
                            paths[i].append(states[i].viewpoint)

        return paths, poses

    def receive(self, init_poses, gold_instructions, instructions, paths, is_valid):
        self.train_init_poses.extend(init_poses)
        self.train_gold_instructions.extend(gold_instructions)
        self.train_instructions.extend(instructions)
        self.train_paths.extend(paths)
        self.train_is_valid.extend(is_valid)

    def process_data(self, model_name, init_poses, instructions, paths, is_valid):

        states = self.init(model_name, init_poses, instructions, False)

        for i in range(self.batch_size):
            if not is_valid[i]:
                self.terminated[i] = True

        t = 0
        teacher_actions = [None] * self.batch_size

        while not self.has_terminated():
            for i in range(self.batch_size):
                if t + 1 < len(paths[i]):
                    next_viewpoint = paths[i][t + 1]
                else:
                    next_viewpoint = paths[i][-1]
                for j, loc in enumerate(states[i].adj_loc_list):
                    if loc['nextViewpointId'] == next_viewpoint:
                        teacher_actions[i] = j
                        break
            self.act(states, teacher_actions=teacher_actions)
            states = states.step(teacher_actions)
            t += 1

    def compute_supervised_loss(self, model_name):

        num_trains = len(self.train_init_poses)

        assert num_trains % self.batch_size == 0

        losses = []

        for i in range(num_trains // self.batch_size):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size

            init_poses = self.train_init_poses[start:end]
            instructions = self.train_instructions[start:end]
            paths = self.train_paths[start:end]
            is_valid = self.train_is_valid[start:end]

            self.process_data(
                model_name, init_poses, instructions, paths, is_valid)

            losses.append(self.compute_loss())

        loss = sum(losses) / len(losses)

        return loss

    def compute_unsupervised_loss(self, model_name):

        init_poses = self.train_init_poses[:self.batch_size]
        instructions = self.train_instructions[:self.batch_size]
        paths = self.world.sample_paths(init_poses)
        is_valid = [1] * self.batch_size

        self.process_data(model_name, init_poses, instructions, paths, is_valid)

        loss = self.compute_loss()

        return loss


    def learn(self, unsup_weight):

        exp_loss = (1 - unsup_weight) * self.compute_supervised_loss('exploration') + \
                    unsup_weight * self.compute_unsupervised_loss('exploration')
        exe_loss = self.compute_supervised_loss('execution')

        loss = exp_loss + exe_loss

        if isinstance(loss, int) or isinstance(loss, float):
            return 0

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if len(self.teacher_action_seqs) == 0:
            return 0

        return loss.item() / len(self.teacher_action_seqs)

    def save(self, name, trajectories=None):
        file_path = '%s/%s' % (self.config.experiment_dir, name + '.ckpt')
        ckpt = { 'exp_model_state_dict': self.exploration_model.state_dict(),
                 'exe_model_state_dict': self.execution_model.state_dict(),
                 'optim_state_dict'    : self.optim.state_dict() }
        torch.save(ckpt, file_path)
        logging.info('Saved %s model to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(file_path, map_location=self.device)
        if 'model_state_dict' in ckpt:
            self.exploration_model.load_state_dict(ckpt['model_state_dict'])
            self.execution_model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.exploration_model.load_state_dict(ckpt['exp_model_state_dict'])
            self.execution_model.load_state_dict(ckpt['exe_model_state_dict'])
        logging.info('Loaded model from %s' % file_path)

        if hasattr(self.config.student, 'not_load_optim') and self.config.student.not_load_optim:
            return

        if 'last.ckpt' in file_path:
            self.optim.load_state_dict(ckpt['optim_state_dict'])
            logging.info('Loaded optim from %s' % file_path)





