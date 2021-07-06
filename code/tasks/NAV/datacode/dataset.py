import os
import sys
import json
import logging
import string
import numpy as np
import random
import re
import ast
from collections import Counter
sys.path.append('..')

from misc import util


class Dataset(object):

    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

    def __init__(self, config, split):

        self.config = config
        self.random = random.Random(self.config.seed)
        self.split = split

        self.data = self.load_data(config.data_dir, config.task, split)

        self.item_idx = 0
        self.batch_size = config.trainer.batch_size

        if split == 'train':
            config.vocab = self.load_vocab(config.data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def _split_sentence(self, sentence):
        toks = []
        words = [s.strip().lower()
            for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip())
                if len(s.strip()) > 0]
        for word in words:
            if all(c in string.punctuation for c in word) and \
               not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def load_data(self, data_dir, task, split):

        data_file = os.path.join(data_dir, task + '_' + split + '.json')
        with open(data_file) as f:
            data = json.load(f)

        for item in data:
            instructions = item['instructions']
            item['instructions'] = []
            for instruction in instructions:
                tokenized_instruction = self._split_sentence(instruction)
                item['instructions'].append(tokenized_instruction)

        logging.info('Loaded %d instances of %s split from %s' %
            (len(data), self.split, data_file))

        return data

    def build_vocab(self, min_count=5):
        count = Counter()
        for item in self.data:
            for instruction in item['instructions']:
                count.update(instruction)
        vocab = []
        for word, num in count.most_common():
            if num >= min_count:
                vocab.append(word)
            else:
                break
        return vocab

    def load_vocab(self, data_dir):

        vocab_file = os.path.join(data_dir, 'vocab.json')
        if os.path.exists(vocab_file):
            with open(vocab_file) as f:
                words = json.load(f)
        else:
            words = self.build_vocab()
            with open(vocab_file, 'w') as f:
                json.dump(words, f, indent=2)

        vocab = util.Vocab()
        for w in words:
            vocab.index(w)

        logging.info('Loaded word vocab of size %d' % len(vocab))

        return vocab

    def finalize_batch(self, batch):

        new_batch = []
        for item in batch:
            new_item = {}
            new_item['instr_id'] = item['path_id']
            new_item['path']    = item['path']
            new_item['scan']    = item['scan']
            new_item['heading'] = item['heading']
            new_item['equiv_instructions'] = item['instructions']
            new_item['instruction'] = self.random.choice(item['instructions'])
            new_batch.append(new_item)

        return new_batch

    def iterate_batches(self, batch_size=None, data_idx=None, data_indices=None):

        if batch_size is None:
            batch_size = self.batch_size

        if data_indices is None:
            self.indices = list(range(len(self.data)))
            if self.split == 'train':
                self.random.shuffle(self.indices)
        else:
            self.indices = data_indices

        assert len(self.indices) == len(self.data)

        if data_idx is None:
            self.idx = 0
        else:
            self.idx = data_idx

        while True:
            start_idx = self.idx
            end_idx = self.idx + batch_size

            self.idx = end_idx
            if self.idx >= len(self.data):
                self.idx = 0

            batch_indices = self.indices[start_idx:end_idx]

            if len(batch_indices) < batch_size:
                batch_indices += self.random.sample(
                    self.indices, batch_size - len(batch_indices))

            batch = [self.data[i] for i in batch_indices]

            yield self.finalize_batch(batch)

            if self.idx == 0 and self.split != 'train':
                break

