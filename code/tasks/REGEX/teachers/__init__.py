from .iliad import IliadTeacher
from .reinforce import ReinforceTeacher
from .dagger import DaggerTeacher


def load(config):
    cls_name = config.teacher.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such teacher: {}".format(cls_name))
