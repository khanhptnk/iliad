import os
import sys
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

import models
import worlds


class Describer(object):

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.vocab = config.vocab

        model_config = config.describer.model
        model_config.device = config.device
        model_config.vocab_size = len(self.vocab)
        model_config.loc_embed_size = config.world.loc_embed_size
        model_config.pad_idx = self.vocab['<PAD>']
        model_config.n_actions = len(self.vocab)

        self.model = models.load(model_config).to(self.device)

        logging.info('model: ' + str(self.model))

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=model_config.learning_rate)

        if hasattr(model_config, 'load_from'):
            self.load(model_config.load_from)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab['<PAD>'])
        self.n_samples = config.describer.n_samples

        self.world = worlds.load(config)

    def _to_tensor(self, x):
        return torch.tensor(x).to(self.device)

    def _to_tensor_from_numpy(self, x):
        return torch.from_numpy(x).to(self.device)

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

    def _view_features_variable(self, states):
        view_features = [state.curr_view_features for state in states]
        view_features = self._to_tensor_from_numpy(np.stack(view_features))

        return view_features

    def _pose_features_variable(self, states):
        pose_features = [state.pose_embedding for state in states]
        pose_features = self._to_tensor_from_numpy(np.stack(pose_features))
        return pose_features

    def _path_features_variable(self, init_poses, paths):

        action_embed_seqs = []
        view_feature_seqs = []

        states = self.world.init(init_poses)
        max_len = max([len(path) for path in paths])
        for i in range(max_len):

            # View features
            view_feature_seqs.append(self._view_features_variable(states))

            # Find action indices
            next_viewpoints = []
            for path in paths:
                if i + 1 < len(path):
                    next_viewpoints.append(path[i + 1])
                else:
                    next_viewpoints.append(path[-1])
            actions = []
            for state, next_viewpoint in zip(states, next_viewpoints):
                for j, loc in enumerate(state.adj_loc_list):
                    if loc['nextViewpointId'] == next_viewpoint:
                        actions.append(j)
                        break

            # Action features
            all_action_embeds, _ = self._nav_action_variable(states)
            action_embeds = all_action_embeds[np.arange(self.batch_size), actions, :]
            action_embed_seqs.append(action_embeds)

            states = states.step(actions)

        path_masks = []
        for path in paths:
            path_masks.append([0] * len(path) + [1] * (max_len - len(path)))
        path_masks = self._to_tensor(path_masks).bool()

        return action_embed_seqs, view_feature_seqs, path_masks

    def init(self, init_poses, paths, is_eval):

        if is_eval:
            self.model.eval()
        else:
            self.model.train()

        self.is_eval = is_eval
        self.batch_size = len(paths)

        self.pred_action_seqs = [[] for _ in range(self.batch_size)]
        self.teacher_action_seqs = []
        self.action_logit_seqs = []
        self.terminated = [False] * self.batch_size

        action_embed_seqs, view_feature_seqs, path_masks = \
            self._path_features_variable(init_poses, paths)

        self.dec_h, self.dec_time, self.path_encodings = self.model.encode(
            action_embed_seqs, view_feature_seqs)
        self.path_masks = path_masks

        self.prev_actions = [self.vocab['<BOS>']] * self.batch_size
        self.prev_actions = self._to_tensor(self.prev_actions).long()

        self.timer = self.config.describer.max_timesteps

    def act(self, teacher_actions=None, sample=False):

        self.dec_h, self.dec_time, action_logits = self.model.decode(
                self.dec_h,
                self.dec_time,
                self.prev_actions,
                self.path_encodings,
                self.path_masks
            )

        self.action_logit_seqs.append(action_logits)

        if self.is_eval:
            if sample:
                pred_actions = D.Categorical(logits=action_logits).sample()
            else:
                pred_actions = action_logits.max(dim=1)[1]
            self.prev_actions = pred_actions
            for i in range(self.batch_size):
                if not self.terminated[i] and pred_actions[i].item() != self.vocab['<EOS>']:
                    self.pred_action_seqs[i].append(
                        self.vocab.get(pred_actions[i].item()))
        else:
            for i in range(self.batch_size):
                if self.terminated[i]:
                    teacher_actions[i] = '<PAD>'
            teacher_actions = [self.vocab[w] for w in teacher_actions]
            teacher_actions = self._to_tensor(teacher_actions).long()
            self.prev_actions = teacher_actions
            self.teacher_action_seqs.append(teacher_actions)

        self.timer -= 1

        for i in range(self.batch_size):
            self.terminated[i] |= self.timer <= 0
            self.terminated[i] |= self.prev_actions[i].item() == self.vocab['<EOS>']

    def has_terminated(self):
        return all(self.terminated)

    def get_action_seqs(self):
        return self.pred_action_seqs

    def predict(self, init_poses, paths, sample=False):

        self.init(init_poses, paths, True)
        while not self.has_terminated():
            self.act(sample=sample)

        return self.get_action_seqs()

    def pragmatic_predict(self, init_poses, paths, executor, teacher):

        def eval(pred_instructions):

            pred_paths, _ = executor.predict(init_poses, pred_instructions)
            pred_goal_viewpoints = [path[-1] for path in pred_paths]

            metrics = []
            for scan, pred_path, gold_path in zip(scans, pred_paths, paths):
                metric = teacher.eval(scan, pred_path, gold_path)
                metrics.append(metric)

            return metrics, pred_paths

        metric_name = self.config.trainer.main_metric_name

        scans = [pose[0] for pose in init_poses]
        goal_viewpoints = [path[-1] for path in paths]

        best_pred_instructions = self.predict(init_poses, paths)
        best_metrics, best_pred_paths = eval(best_pred_instructions)

        for _ in range(self.n_samples):

            pred_instructions = self.predict(init_poses, paths, sample=True)
            metrics, pred_paths = eval(pred_instructions)

            for i in range(self.batch_size):
                best = best_metrics[i][metric_name]
                cand = metrics[i][metric_name]
                if teacher.is_better(metric_name, cand, best):
                    best_metrics[i] = metrics[i]
                    best_pred_instructions[i] = pred_instructions[i]
                    best_pred_paths[i] = pred_paths[i]

        return best_pred_instructions, best_pred_paths, best_metrics

    def compute_loss(self):
        loss = 0
        zipped_info = zip(self.action_logit_seqs, self.teacher_action_seqs)
        for logits, refs in zipped_info:
            loss += self.loss_fn(logits, refs)
        return loss

    def learn(self):

        assert len(self.teacher_action_seqs) == len(self.action_logit_seqs)

        loss = self.compute_loss()

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





