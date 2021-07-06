import os
import sys
import time
import argparse
import logging
import numpy as np
from datetime import datetime

import torch

import flags
import datacode
import trainers
import agents
import teachers

from misc import util


def main():

    config = configure()
    datasets = datacode.load(config)
    trainer = trainers.load(config)
    student = agents.load(config)
    teacher = teachers.load(config)

    with torch.cuda.device(config.device_id):
        trainer.train(datasets, student, teacher)

def configure():

    config = flags.make_config()

    config.command_line = 'python3 -u ' + ' '.join(sys.argv)

    config.data_dir = '%s/%s' % (config.data_path, config.task)
    output_dir = '%s/experiments' % config.save_path

    print('Data directory is ', config.data_dir)
    print('Output direction is ', output_dir)

    config.experiment_dir = '%s/%s' % (output_dir, config.name)

    if config.resume:
        ckpt_file = '%s/%s' % (config.experiment_dir, 'last.ckpt')
        if os.path.exists(ckpt_file):
            print('Resume from %s' % ckpt_file)
            config.student.model.load_from = ckpt_file
        elif not os.path.exists(config.experiment_dir):
            os.makedirs(config.experiment_dir)
    else:
        assert not os.path.exists(config.experiment_dir), \
            'Experiment %s already exists!' % config.experiment_dir
        os.makedirs(config.experiment_dir)

    torch.manual_seed(config.seed)
    random = np.random.RandomState(config.seed)
    config.random = random

    config.device = torch.device('cuda', config.device_id)

    config.start_time = time.time()

    log_file = '%s/run.log' % config.experiment_dir
    util.config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info(config.command_line)
    logging.info('Write log to %s' % log_file)
    logging.info(str(config))

    return config

if __name__ == '__main__':
    main()
