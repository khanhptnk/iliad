import os
import sys
import json
import logging
import string
import numpy as np
import random
sys.path.append('..')

from misc import util


class Dataset(object):

    N_EXAMPLES = 5

    def __init__(self, config, split):

        self.config = config
        self.random = random.Random(self.config.seed)
        self.split = split

        self.file_name = os.path.join(config.data_dir, config.data_file)
        self.data = self.load_data(self.file_name, split)

        self.item_idx = 0
        self.batch_size = config.trainer.batch_size

        if split == 'train':
            config.word_vocab, config.char_vocab, config.regex_vocab = \
                self.load_vocab(config.data_dir)
            logging.info('Constructed word vocab of size %d' %
                len(config.word_vocab))
            logging.info('Constructed char vocab of size %d' %
                len(config.char_vocab))
            logging.info('Constructed regex vocab of size %d' %
                len(config.regex_vocab))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def load_data(self, file_name, split):
        with open(file_name) as f:
            data = json.load(f)[split]
        logging.info('Loaded %d instances of %s split from %s' %
            (len(data), self.split, file_name))
        return data

    def load_vocab(self, data_dir):

        vocab = {
            'word': util.Vocab(),
            'char': util.Vocab(),
            'regex': util.Vocab()
        }

        vocab_file = os.path.join(data_dir, 'vocab.json')
        with open(vocab_file) as f:
            data = json.load(f)

        for name in vocab:
            for w in data[name]:
                vocab[name].index(w)

        return vocab['word'], vocab['char'], vocab['regex']

    def finalize_batch(self, batch):

        new_batch = []
        for item in batch:

            regex = item['re']
            examples = item['examples']

            if self.split == 'train':
                #src_word, tgt_word = self.random.choice(examples)
                examples = self.random.sample(examples, self.N_EXAMPLES)
                src_word, tgt_word = self.random.choice(examples)
            else:
                for src_word, tgt_word in examples:
                    if src_word != tgt_word:
                        break

            new_item = {
                    'regex': regex,
                    'instruction' : self.random.choice(item['instructions']),
                    'src_word': src_word,
                    'tgt_word': tgt_word,
                    'examples': examples,
                    'equiv_instructions': item['instructions']
                }

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

            batch = [self.data[i] for i in batch_indices]

            yield self.finalize_batch(batch)

            if self.idx == 0 and self.split != 'train':
                break

