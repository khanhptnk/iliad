from .executor import ExecutorTrainer
from .describer import DescriberTrainer
from .iliad import IliadTrainer
from .reinforce import ReinforceTrainer
from .dagger import DaggerTrainer


def load(config):
    cls_name = config.trainer.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such trainer: {}".format(cls_name))
