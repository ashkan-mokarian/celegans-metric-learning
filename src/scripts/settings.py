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


class Path(BaseSettings):
    def __init__(self, base_path):
        self.BASE = base_path
        self.DATA = get_abs_join_path(base_path, 'data')
        self.DATA_INTERIM = get_abs_join_path(self.DATA, 'interim')
        proc_data = get_abs_join_path(base_path, 'data', 'processed')
        self.CPM_DATASET = get_abs_join_path(proc_data, 'cpm_dataset.pkl')
        self.WORMS_DATASET = get_abs_join_path(proc_data, 'worms_dataset')

        self.EXPERIMENTS = get_abs_join_path(base_path, 'experiments')

        self.EXPERIMENTS_CFG = get_abs_join_path(base_path, 'experiments_cfg')


class General(BaseSettings):
    def __init__(self):
        self.LOGGING = 20
        self.OVERWRITE = False
        self.DEBUG = False


class Settings(BaseSettings):
    """Populates setting values (path, data params, train params, etc) with defaults values. overwrites the ones
    available in the provided config files"""
    def __init__(self, name=None, confs=None):
        base_path = get_abs_join_path(
            __file__,
            os.path.pardir,
            os.path.pardir,
            os.path.pardir
            )
        self.NAME = name
        self.PATH = Path(base_path=base_path)
        self.GENERAL = General()

        self.read_confs(confs)

        # overwrite given name, instead of reading it from conf
        if name:
            self.NAME = name

    def read_confs(self, confs=None):
        if confs is None:
            logger.debug("confs value from cli input is None")
            pass
        if not isinstance(confs, list):
            confs = [confs]
        for conf in confs:
            if os.path.isfile(conf):
                self._read_conf(conf)
            else:
                found_conf = []
                for r, d, f in os.walk(self.PATH.EXPERIMENTS_CFG):
                    for file in f:
                        if file == conf +'.toml' or file == conf:
                            found_conf.append(get_abs_join_path(r, file))
                assert len(found_conf) == 1,\
                    f'Foudn too many config files with the same name {conf} in {self.PATH.EXPERIMENTS_CFG}'
                _replace_attrs_with_dict(self, toml.load(found_conf[0]))


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
    sett = Settings('Cat', confs='train_default')
    print(sett)
    print('Finished')