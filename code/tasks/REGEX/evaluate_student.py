import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch

import flags
import datacode
import trainers
import agents

from misc import util


def main():

    config = configure()
    datasets = datacode.load(config)
    trainer = trainers.load(config)
    student = agents.load(config)

    with torch.cuda.device(config.device_id):
        trainer.evaluate(datasets['val'], student, save_pred=True)
        trainer.evaluate(datasets['test'], student, save_pred=True)

def configure():

    config = flags.make_config()

    config.command_line = 'python3 -u ' + ' '.join(sys.argv)

    assert os.path.exists(config.student.model.load_from), \
            "Experiment %s not exists!" % config.student.model.load_from

    config.experiment_dir = os.path.join(*config.student.model.load_from.split('/')[:-1])

    torch.manual_seed(config.seed)
    random = np.random.RandomState(config.seed)
    config.random = random

    config.device = torch.device('cuda', config.device_id)

    config.start_time = time.time()

    log_file = os.path.join(config.experiment_dir, 'eval.log')
    util.config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info(config.command_line)
    logging.info(str(config))

    return config

if __name__ == '__main__':
    main()
