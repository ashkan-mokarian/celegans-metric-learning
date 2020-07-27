class Worms(object):

    def __init__(self, file_path):
        self._worm_names = Worms._get_worm_names(file_path)
        self._name_to_uid = None
        self._uid_to_name = None

    def _get_worm_names(file_path):
        worm_names = list()
        with open(file_path) as f:
            for line in f:
                worm_names.append(line.strip())
        return worm_names

    def _get_name_to_uid_map(self):
        name_to_uid = dict()
        name_to_uid.update([(name, uid) for uid, name in enumerate(self._worm_names, start=1)])
        return name_to_uid

    def _get_uid_to_name_map(self):
        uid_to_name = dict()
        uid_to_name.update([(uid, name) for uid, name in enumerate(self._worm_names, start=1)])
        return uid_to_name

    def name_to_uid(self, label):
        if self._name_to_uid is None:
            self._name_to_uid = self._get_name_to_uid_map()
        return self._name_to_uid[label]

    def uid_to_name(self, uid):
        if self._uid_to_name is None:
            self._uid_to_name = self._get_uid_to_name_map()
        return self._uid_to_name[uid]

    def is_valid_name(self, name):
        return name.upper() in self._worm_names