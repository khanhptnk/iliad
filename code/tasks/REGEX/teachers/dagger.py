import os
import sys
sys.path.append('..')
import random


class DaggerTeacher(object):

    def __init__(self, config):
        self.config = config

    def receive_simulation_data(self, batch):
        self.gold_regexes = []
        for item in batch:
            self.gold_regexes.append(item['regex'])

    def demonstrate(self, src_words, instructions):
        return self.gold_regexes




