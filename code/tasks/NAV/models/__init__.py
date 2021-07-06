from .executor_transformer_seq2seq import ExecutorTransformerSeq2SeqModel
from .describer_transformer_seq2seq import DescriberTransformerSeq2SeqModel


def load(config):
    cls_name = config.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
