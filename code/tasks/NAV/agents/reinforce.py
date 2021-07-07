import os
import sys
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import models
import worlds


class ReinforceStudent(object):

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

        self.model = models.load(model_config).to(self.device)

        logging.info('model: ' + str(self.model))

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=model_config.learning_rate)

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.max_instruction_length = config.student.max_instruction_length

        self.baseline = [0, 0]

    def _to_tensor(self, x):
        return torch.tensor(x).to(self.device)

    def _to_tensor_from_numpy(self, x):
        return torch.from_numpy(x).to(self.device)

    def _index_and_pad(self, xs, vocab, reverse=True):

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

    def init(self, init_poses, instructions, is_eval):

        if is_eval:
            self.model.eval()
        else:
            self.model.train()

        self.is_eval = is_eval
        self.batch_size = len(instructions)

        self.state_seqs = []
        self.pred_action_seqs = [[] for _ in range(self.batch_size)]
        self.teacher_action_seqs = []
        self.action_logit_seqs = []
        self.logit_mask_seqs = []
        self.terminated = [False] * self.batch_size

        instr_encodings, instr_masks = self._index_and_pad(
            instructions, self.vocab)
        self.text_dec_h, self.state_dec_h, self.dec_time, self.instructions = \
            self.model.encode(instr_encodings, instr_masks)
        self.instruction_masks = instr_masks
        self.prev_action_embeds = self.model.init_action(self.batch_size)

        self.timer = self.config.student.max_timesteps

        init_states = self.world.init(init_poses)

        return init_states

    def act(self, states, teacher_actions=None, bc=False, sample=False):

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
        self.logit_mask_seqs.append(logit_masks)
        self.state_seqs.append(states)

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
            if bc:
                pred_actions = teacher_actions
            else:
                pred_actions = D.Categorical(logits=action_logits).sample().tolist()
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

    def has_terminated(self):
        return all(self.terminated)

    def get_action_seqs(self):
        return self.pred_action_seqs

    def predict(self, init_poses, instructions, sample=False):

        with torch.no_grad():
            states = self.init(init_poses, instructions, True)
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

    def receive(self, init_poses, instructions, pred_paths, rewards):

        assert len(init_poses) == len(rewards)

        batch_size = len(rewards)

        self.baseline[0] += sum(rewards)
        self.baseline[1] += len(rewards)

        self.rewards = self._to_tensor(rewards).float()

        states = self.init(init_poses, instructions, False)

        t = 0
        while not self.has_terminated():
            actions = [None] * batch_size
            for i, state in enumerate(states):
                if t + 1 >= len(pred_paths[i]):
                    actions[i] = self.STOP
                else:
                    for j, loc in enumerate(state.adj_loc_list):
                        if loc['nextViewpointId'] == pred_paths[i][t + 1]:
                            actions[i] = j
                            break

            self.act(states, teacher_actions=actions, bc=True)
            states = states.step(actions)
            t += 1

    def compute_loss(self, entropy_weight):

        assert len(self.teacher_action_seqs) == len(self.action_logit_seqs)

        loss = 0
        zipped_info = zip(self.action_logit_seqs,
                          self.teacher_action_seqs)

        for logits, refs in zipped_info:

            valid = (refs != -1)
            normalizer = valid.sum()

            baseline = self.baseline[0] / self.baseline[1]
            norm_rewards = self.rewards - baseline

            losses = self.loss_fn(logits, refs) * norm_rewards
            # Entropy regularization
            d = D.Categorical(logits.softmax(dim=1))
            entropies = d.entropy()

            losses = losses - entropy_weight * entropies

            losses = losses * valid
            if normalizer > 0:
                loss += losses.sum() / normalizer

        return loss

    def learn(self, entropy_weight):

        loss = self.compute_loss(entropy_weight)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item() / len(self.teacher_action_seqs)

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





