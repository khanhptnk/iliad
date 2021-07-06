import os
import sys
import argparse
import csv
import math
import json
import random
import logging
from collections import defaultdict

import networkx as nx
import numpy as np
import torch

from misc import util
import MatterSim


csv.field_size_limit(sys.maxsize)

angle_inc = np.pi / 6.
NUM_VIEWS = 36
MEAN_POOLED_DIM = 2048

IMAGE_W = 640
IMAGE_H = 480
VFOV = 60


def _build_loc_embedding(heading, elevation, size):
    embedding = np.zeros(size, dtype=np.float32)
    shift = size // 4
    embedding[0          :shift]       = math.sin(heading)
    embedding[shift      :(shift * 2)] = math.cos(heading)
    embedding[(shift * 2):(shift * 3)] = math.sin(elevation)
    embedding[(shift * 3):]            = math.cos(elevation)
    return embedding

def _build_action_embedding(adj_loc_list, features, loc_embed_size):
    feature_dim = features.shape[-1]
    embedding = np.zeros((len(adj_loc_list), feature_dim + loc_embed_size),
        dtype=np.float32)
    for a, adj_dict in enumerate(adj_loc_list):
        if a == 0:
            # the embedding for the first action ('stop') is left as zero
            continue
        embedding[a, :feature_dim] = features[adj_dict['absViewIndex']]
        rel_heading = adj_dict['rel_heading']
        rel_elevation = adj_dict['rel_elevation']
        embedding[a, feature_dim:(feature_dim + loc_embed_size)] = _build_loc_embedding(
            rel_heading, rel_elevation, loc_embed_size)

    return embedding

def _build_view_loc_embedding(loc_embed_size):
    embedding = np.zeros((NUM_VIEWS, loc_embed_size), dtype=np.float32)
    for i in range(NUM_VIEWS):
        heading = (i % 12) * angle_inc
        elevation = ((i // 12) - 1) * angle_inc
        embedding[i, :] = _build_loc_embedding(
            heading, elevation, loc_embed_size)
    return embedding

def _calculate_headings_and_elevations_for_views(sim, goalViewIndices):
    states = sim.getState()
    heading_deltas = []
    elevation_deltas = []
    for state, goalViewIndex in zip(states, goalViewIndices):
        currViewIndex = state.viewIndex
        heading_deltas.append(goalViewIndex % 12 - currViewIndex % 12)
        elevation_deltas.append(goalViewIndex // 12 - currViewIndex // 12)
    return heading_deltas, elevation_deltas


class ImageFeatures(object):

    def __init__(self, image_feature_file, loc_embed_size):

        logging.info('Loading image features from %s' % image_feature_file)
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']

        default_features = np.zeros(
            (NUM_VIEWS, MEAN_POOLED_DIM), dtype=np.float32)
        self.features = defaultdict(lambda: default_features)

        with open(image_feature_file, "rt") as tsv_in_file:
            reader = csv.DictReader(
                tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
            for item in reader:

                assert int(item['image_h']) == IMAGE_H
                assert int(item['image_w']) == IMAGE_W
                assert int(item['vfov']) == VFOV

                long_id = item['scanId'] + '_' + item['viewpointId']
                features = np.frombuffer(util.decode_base64(item['features']),
                    dtype=np.float32).reshape((NUM_VIEWS, MEAN_POOLED_DIM))
                self.features[long_id] = features

    def get_features(self, scan, viewpoint):
        long_id = scan + '_' + viewpoint
        return self.features[long_id]


class MatterportWorldMeta(object):

    def __init__(self, config):

        self.config = config

        self.load_graphs()
        self.load_cached_adjacent_lists()
        self.load_image_features()

    def load_graphs(self):
        scan_list_file = os.path.join(
            self.config.data_dir, 'connectivity/scans.txt')
        scans = set(open(scan_list_file).read().strip().split('\n'))
        self.graphs = {}
        self.paths = {}
        self.distances = {}
        for scan in scans:
            graph = util.load_nav_graphs(scan, self.config.data_dir)
            self.graphs[scan] = graph
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(graph))
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(graph))

    def load_cached_adjacent_lists(self):
        cached_action_space_path = os.path.join(
            self.config.data_dir, 'panoramic_action_space.json')
        with open(os.path.join(cached_action_space_path)) as f:
            self.cached_adj_loc_lists = json.load(f)

    def load_image_features(self):
        image_feature_path = os.path.join(self.config.data_dir, 'img_features',
            self.config.world.image_feature_file)
        self.featurizer = ImageFeatures(
            image_feature_path, self.config.world.loc_embed_size)


class MatterportWorld(object):

    def __init__(self, meta):

        self.config = meta.config

        self.featurizer = meta.featurizer
        self.graphs = meta.graphs
        self.paths = meta.paths
        self.distances = meta.distances
        self.cached_adj_loc_lists = meta.cached_adj_loc_lists

        self.init_simulator()
        self.loc_embed_size = self.config.world.loc_embed_size

        self.random = random.Random(self.config.seed)

    def init(self, poses):
        self.sim.newEpisode(*list(zip(*poses)))
        return MatterportState(self)

    def init_simulator(self):
        self.sim_batch_size = self.config.trainer.batch_size
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setCameraResolution(IMAGE_W, IMAGE_H)
        self.sim.setCameraVFOV(math.radians(VFOV))
        self.sim.setNavGraphPath(
            os.path.join(self.config.data_dir, 'connectivity'))
        self.sim.setBatchSize(self.sim_batch_size)
        self.sim.initialize()

    def eval(self, scan, gold, pred):
        d = self.get_shortest_distance(scan, gold, pred)
        return d, d <= self.success_radius

    def get_shortest_distance(self, scan, start, end):
        return self.distances[scan][start][end]

    def get_shortest_path(self, scan, start, end):
        return self.paths[scan][start][end]

    def get_path_length(self, scan, path):
        length = 0
        for i, v in enumerate(path[1:]):
            u = path[i]
            length += self.get_shortest_distance(scan, u, v)
        return length

    def get_node_position(self, scan, node):
        return self.graphs[scan].nodes[node]['position']

    def sample_paths(self, poses):
        paths = []
        for pose in poses:
            scan, viewpoint = pose[0], pose[1]
            paths.append(self.sample_path(scan, viewpoint))

        return paths

    def sample_path(self, scan, start, min_nodes=3, max_nodes=7):
        paths = list(self.paths[scan][start].values())
        num_nodes = self.random.randint(min_nodes, max_nodes)
        path = self.random.choice(paths)[:num_nodes]
        return path


class MatterportState(object):

    def __init__(self, world):

        self.world = world

        self.states = []

        for sim_state in world.sim.getState():

            state = util.Struct()
            state.scan      = sim_state.scanId
            state.viewpoint = sim_state.location.viewpointId
            state.view_id   = sim_state.viewIndex
            state.heading   = sim_state.heading
            state.elevation = sim_state.elevation
            state.location  = sim_state.location

            long_id = '_'.join(
                [state.scan, state.viewpoint, str(state.view_id % 12)])
            state.adj_loc_list = world.cached_adj_loc_lists[long_id]

            state.curr_view_features = world.featurizer.get_features(
                state.scan, state.viewpoint)

            state.action_embeddings = _build_action_embedding(
                state.adj_loc_list, state.curr_view_features,
                world.loc_embed_size)

            self.states.append(state)

    def __getitem__(self, i):
        return self.states[i]

    def __iter__(self):
        return iter(self.states)

    def step(self, actions):

        navigable_view_indices = []
        next_viewpoint_ids = []
        next_view_indices = []

        for state, action in zip(self.states, actions):
            loc = state.adj_loc_list[action]
            next_viewpoint_ids.append(loc['nextViewpointId'])
            next_view_indices.append(loc['absViewIndex'])
            navigable_view_indices.append(loc['absViewIndex'])

        new_states = self._navigate_to_locations(next_viewpoint_ids,
            next_view_indices, navigable_view_indices)

        return new_states

    def _navigate_to_locations(self,
            next_viewpoint_ids, next_view_indices, navigable_view_indices):

        sim = self.world.sim

        # Rotate to the view index assigned to the next viewpoint
        heading_deltas, elevation_deltas = \
            _calculate_headings_and_elevations_for_views(
                sim, navigable_view_indices)

        sim.makeAction(
            [0] * len(heading_deltas), heading_deltas, elevation_deltas)

        states = sim.getState()
        locationIds = []

        assert len(states) == len(next_viewpoint_ids) == len(navigable_view_indices)

        zipped_info = zip(states,
                          next_viewpoint_ids,
                          navigable_view_indices)

        for i, (state, next_viewpoint_id, navigable_view_index) in enumerate(zipped_info):

            # Check if rotation was done right
            assert state.viewIndex == navigable_view_index

            # Find index of the next viewpoint
            index = None
            for i, loc in enumerate(state.navigableLocations):
                if loc.viewpointId == next_viewpoint_id:
                    index = i
                    break
            assert index is not None, state.scanId + ' ' + state.location.viewpointId + ' ' + next_viewpoint_id
            locationIds.append(index)

        # Rotate to the target view index
        heading_deltas, elevation_deltas = \
            _calculate_headings_and_elevations_for_views(sim, next_view_indices)

        sim.makeAction(locationIds, heading_deltas, elevation_deltas)

        # Final check
        states = sim.getState()
        zipped_info = zip(states, next_viewpoint_ids, next_view_indices)
        for state, next_viewpoint_id, next_view_index in zipped_info:
            assert state.viewIndex == next_view_index
            assert state.location.viewpointId == next_viewpoint_id

        return MatterportState(self.world)


