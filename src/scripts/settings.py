import os
import logging
import toml

logger = logging.getLogger(__name__)


def get_abs_join_path(*p):
    return os.path.abspath(os.path.join(*p))


class BaseSettings:
    def __init__(self):
        raise NotImplementedError

    def __str__(self, depth=0):
        s = '\t'*depth + '\n'
        for k, v in self.__dict__.items():
            s += '\t'*depth + '[' + str(k) + ']: '
            if issubclass(type(v), BaseSettings):
                s += v.__str__(depth=depth+1)
            else:
                s += str(v) + '\n'
        # s += '\t'*depth + '\n'
        return s

    def read_confs(self, confs=None):
        if confs is None:
            logger.debug("confs value from cli input is None")
            pass
        if not isinstance(confs, list):
            confs = [confs]
        for conf in confs:
            if os.path.isfile(conf):
                _replace_attrs_with_dict(self, toml.load(conf))
            else:
                found_conf = []
                for r, d, f in os.walk(DEFAULT_PATH.EXPERIMENTS_CFG):
                    for file in f:
                        if file == conf +'.toml' or file == conf:
                            found_conf.append(get_abs_join_path(r, file))
                assert len(found_conf) == 1,\
                    f'Foudn too many config files with the same name {conf} in {DEFAULT_PATH.EXPERIMENTS_CFG}'
                _replace_attrs_with_dict(self, toml.load(found_conf[0]))

    def get_toml_dict(self, filename=None):
        toml_dict = {}
        for attr, attrval in self.__dict__.items():
            if isinstance(attrval, BaseSettings):
                toml_dict.update({attr: attrval.get_toml_dict()})
            else:
                toml_dict.update({attr:attrval})
        if filename:
            with open(filename, 'w') as f:
                toml.dump(toml_dict, f=f)
        return toml_dict


class DefaultPath(BaseSettings):
    def __init__(self):
        self.BASE = get_abs_join_path(
            __file__,
            os.path.pardir,
            os.path.pardir,
            os.path.pardir
            )
        self.DATA = get_abs_join_path(self.BASE, 'data')
        self.DATA_INTERIM = get_abs_join_path(self.DATA, 'interim')
        proc_data = get_abs_join_path(self.BASE, 'data', 'processed')
        self.EXPERIMENTS = get_abs_join_path(self.BASE, 'experiments')
        self.EXPERIMENTS_CFG = get_abs_join_path(self.BASE, 'experiments_cfg')
        self.WORM_NAMES = get_abs_join_path(self.DATA, 'raw', 'worm_names.txt')
DEFAULT_PATH = DefaultPath()


class Path(BaseSettings):
    def __init__(self):
        self.EXPERIMENT_ROOT = None
        self.WORMS_DATASET = None
        self.CPM_DATASET = None


class General(BaseSettings):
    def __init__(self):
        self.LOGGING = 20
        self.OVERWRITE = False
        self.DEBUG = False
        self.SEED = None
        self.GPU_DEBUG = None
        self.GPU_DEBUG_TRACE_INTO = None


class Model(BaseSettings):
    def __init__(self):
        self.MODEL_NAME = None
        self.MODEL_PARAMS = None
        self.INIT_MODEL_PATH = None
        self.INIT_MODEL_BEST = False
        self.INIT_MODEL_LAST = False
        self.PADDING = False


class Train(BaseSettings):
    def __init__(self):
        self.N_CLUSTER = None
        self.N_STEP = None
        self.MODEL_CKPT_EVERY_N_STEP = None
        self.RUNNING_LOSS_INTERVAL = None
        self.BURN_IN_STEP = None
        self.LEARNING_RATE = None
        self.WEIGHT_DECAY = None
        self.LR_DROP_FACTOR = None
        self.LR_DROP_PATIENCE = None
        self.AUGMENTATION = Augmentation()


class Augmentation(BaseSettings):
    def __init__(self):
        self.ELASTIC = Elastic()


class Elastic(BaseSettings):
    def __init__(self):
        self.CONTROL_POINT_SPACING = None
        self.JITTER_SIGMA = None
        self.ROTATION_INTERVAL = None
        self.SUBSAMPLE = None
        self.P = None


class Data(BaseSettings):
    def __init__(self):
        self.N_WORKER = None
        self.PATCH_SIZE = None
        self.OUTPUT_SIZE = None
        self.N_CONSISTENT_WORMS = None
        self.USE_LEFTOUT_LABELS = None
        self.USE_COORD = None
        self.NORMALIZE = None
        self.MAX_NINSTANCE = None


class Settings(BaseSettings):
    """Populates setting values (path, data params, train params, etc) with defaults values. overwrites the ones
    available in the provided config files"""
    def __init__(self, confs=None):
        self.NAME = None
        self.PATH = Path()
        self.GENERAL = General()
        self.MODEL = Model()
        self.TRAIN = Train()
        self.DATA = Data()
        self.read_confs(confs)


def _replace_attrs_with_dict(obj, conf):
    for k, v in conf.items():
        if isinstance(v, dict):
            _replace_attrs_with_dict(getattr(obj, k.upper()), v)
        else:
            if hasattr(obj, k.upper()):
                setattr(obj, k.upper(), v)
            else:
                logger.error(f'Tried to set attribute [{k.upper()}] to [{v}], but class [{obj.__class__}] does not have the attribute.')
                raise ValueError


if __name__ == '__main__':
    sett = Settings(confs='train_default')
    a = sett.get_toml_dict()
    print('Finished')