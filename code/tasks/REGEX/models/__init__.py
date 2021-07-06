from .describer_lstm_seq2seq import DescriberLSTMSeq2SeqModel
from .executor_lstm_seq2seq import ExecutorLSTMSeq2SeqModel
from .student_lstm_seq2seq import StudentLSTMSeq2SeqModel

def load(config):
    cls_name = config.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such model: {}".format(cls_name))
