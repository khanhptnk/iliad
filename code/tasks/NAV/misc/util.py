import re
import os
import math
import time
import logging
import sys
import base64
import json
import numpy as np
import networkx as nx
from datetime import datetime


class Struct:
    def __init__(self, **entries):
        rec_entries = {}
        for k, v in entries.items():
            if isinstance(v, dict):
                rv = Struct(**v)
            elif isinstance(v, list):
                rv = []
                for item in v:
                    if isinstance(item, dict):
                        rv.append(Struct(**item))
                    else:
                        rv.append(item)
            else:
                rv = v
            rec_entries[k] = rv
        self.__dict__.update(rec_entries)

    def __str_helper(self, depth):
        lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, Struct):
                v_str = v.__str_helper(depth + 1)
                lines.append("%s:\n%s" % (k, v_str))
            else:
                lines.append("%s: %r" % (k, v))
        indented_lines = ["    " * depth + l for l in lines]
        return "\n".join(indented_lines)

    def __str__(self):
        return "struct {\n%s\n}" % self.__str_helper(1)

    def __repr__(self):
        return "Struct(%r)" % self.__dict__


class Index:
    def __init__(self):
        self.contents = dict()
        self.ordered_contents = []
        self.reverse_contents = dict()

    def __getitem__(self, item):
        if item not in self.contents:
            return None
        return self.contents[item]

    def index(self, item):
        if item not in self.contents:
            idx = len(self.contents) + 1
            self.ordered_contents.append(item)
            self.contents[item] = idx
            self.reverse_contents[idx] = item
        idx = self[item]
        assert idx != 0
        return idx

    def get(self, idx):
        if idx == 0:
            return "*invalid*"
        return self.reverse_contents[idx]

    def __len__(self):
        return len(self.contents) + 1

    def __iter__(self):
        return iter(self.ordered_contents)


class Vocab(Index):

    def __init__(self):

        super(Vocab, self).__init__()
        self.index('<PAD>')
        self.index('<UNK>')
        self.index('<BOS>')
        self.index('<EOS>')

    def __getitem__(self, item):
        if item not in self.contents:
            return self.contents['<UNK>']
        return self.contents[item]


class ElapsedFormatter():

    def __init__(self):
        self.start_time = datetime.now()

    def format_time(self, t):
        return str(t)[:-7]

    def format(self, record):
        elapsed_time = self.format_time(datetime.now() - self.start_time)
        log_str = "%s %s: %s" % (elapsed_time,
                                record.levelname,
                                record.getMessage())

        return log_str


def config_logging(log_file):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ElapsedFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(ElapsedFormatter())

    logging.basicConfig(level=logging.INFO,
                        handlers=[stream_handler, file_handler])

    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = handler


def flatten(lol):
    if isinstance(lol, tuple) or isinstance(lol, list):
        return sum([flatten(l) for l in lol], [])
    else:
        return [lol]

def postorder(tree):
    if isinstance(tree, tuple):
        for subtree in tree[1:]:
            for node in postorder(subtree):
                yield node
        yield tree[0]
    else:
        yield tree

def tree_map(function, tree):
    if isinstance(tree, tuple):
        head = function(tree)
        tail = tuple(tree_map(function, subtree) for subtree in tree[1:])
        return (head,) + tail
    return function(tree)

def tree_zip(*trees):
    if isinstance(trees[0], tuple):
        zipped_children = [[t[i] for t in trees] for i in range(len(trees[0]))]
        zipped_children_rec = [tree_zip(*z) for z in zipped_children]
        return tuple(zipped_children_rec)
    return trees

FEXP_RE = re.compile(r"(.*)\[(.*)\]")
def parse_fexp(fexp):
    m = FEXP_RE.match(fexp)
    return (m.group(1), m.group(2))

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def add_stat(a, b):
    return (a[0] + sum(b), a[1] + len(b))

def decode_base64(string):
    if sys.version_info[0] == 2:
        return base64.decodestring(bytearray(string))
    elif sys.version_info[0] == 3:
        return base64.decodebytes(bytearray(string, 'utf-8'))
    else:
        raise ValueError("decode_base64 can't handle python version {}".format(sys.version_info[0]))

def load_nav_graphs(scan, data_dir):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    scan_file = os.path.join(data_dir, 'connectivity/%s_connectivity.json' % scan)
    with open(scan_file) as f:
        G = nx.Graph()
        positions = {}
        data = json.load(f)
        for i, item in enumerate(data):
            if item['included']:
                for j, conn in enumerate(item['unobstructed']):
                    if conn and data[j]['included']:
                        positions[item['image_id']] = np.array([item['pose'][3],
                                item['pose'][7], item['pose'][11]]);
                        assert data[j]['unobstructed'][i], 'Graph should be undirected'
                        G.add_edge(item['image_id'],data[j]['image_id'],
                            weight=distance(item,data[j]))
        nx.set_node_attributes(G, values=positions, name='position')
        return G
