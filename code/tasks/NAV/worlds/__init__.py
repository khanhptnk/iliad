from .matterport import MatterportWorldMeta, MatterportWorld

def load_meta(config):
    cls_name = config.world.meta_name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such world meta: {}".format(cls_name))

def load(config):
    cls_name = config.world.name
    try:
        cls = globals()[cls_name]
        try:
            meta = globals()['meta']
        except KeyError:
            meta = load_meta(config)
            globals()['meta'] = meta
        return cls(meta)
    except KeyError:
        raise Exception("No such world: {}".format(cls_name))

