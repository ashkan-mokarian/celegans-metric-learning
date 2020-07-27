"""Note: label uid starts from 1 and not 0, since in most setups, 0 is reserved for background labels"""

class Labels(object):

    def __init__(self, file_path):
        self._labels = Labels._get_labels(file_path)
        self._label_to_uid = None
        self._uid_to_label = None

    def _get_labels(file_path):
        labels = set()
        with open(file_path) as f:
            for line in f:
                labels.add(line.strip().upper())
        return labels

    def _get_label_to_uid_map(self):
        label_to_uid = dict()
        label_to_uid.update([(label, uid+1) for uid, label in enumerate(self._labels)])
        return label_to_uid

    def _get_uid_to_label_map(self):
        uid_to_label = dict()
        uid_to_label.update([(uid+1, label) for uid, label in enumerate(self._labels)])
        return uid_to_label

    def label_to_uid(self, label):
        if self._label_to_uid is None:
            self._label_to_uid = self._get_label_to_uid_map()
        return self._label_to_uid[label]

    def uid_to_label(self, uid):
        if self._uid_to_label is None:
            self._uid_to_label = self._get_uid_to_label_map()
        return self._uid_to_label[uid]

    def is_valid_label(self, label: str):
        if self._labels:
            return label.upper() in self._labels
        else:
            return ValueError
