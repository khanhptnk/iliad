import jsonargparse
import yaml
import numpy as np

from misc.util import Struct


def update_config(source, target):
    for k in source.keys():
        if isinstance(source[k], dict):
            if k not in target:
                target[k] = {}
            update_config(source[k], target[k])
        elif source[k] is not None:
            target[k] = source[k]


def make_config():

    parser = jsonargparse.ArgumentParser()

    parser.add_argument('-config_file', type=str)

    parser.add_argument('--data_path', default='../../../data', help='folder for reading the data')
    parser.add_argument('--save_path', default='.', help='folder for saving the results')
    parser.add_argument('-resume', type=int, default=0)

    parser.add_argument('-student.model.load_from', type=str)
    parser.add_argument('-executor.model.load_from', type=str)
    parser.add_argument('-describer.model.load_from', type=str)

    parser.add_argument('-seed', type=int)
    parser.add_argument('-name', type=str)
    parser.add_argument('-data_dir', type=str, default='../../../data/regex')

    flags = parser.parse_args()

    with open(flags.config_file) as f:
        config = yaml.safe_load(f)

    update_config(jsonargparse.namespace_to_dict(flags), config)

    config = Struct(**config)

    return config


if __name__ == '__main__':
    config = make_config()
    print(config)
