from .describer import Describer
from .executor import Executor
from .iliad import IliadStudent
from .reinforce import ReinforceStudent
from .dagger import DaggerStudent


def load(config, agent_type='student'):

    cls_name = getattr(config, agent_type).name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))

