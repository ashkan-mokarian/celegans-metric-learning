"""Utility functions for parsing useful data from text format outputs"""
import logging

logger = logging.getLogger(__name__)


def read_pm_sol_listfrmt(file):
    pm = dict()
    with open(file) as f:
        target_labels = f.readline()
    target_labels = target_labels[1:-1].split(',')
    pm.update({int(k):int(v) for k, v in enumerate(target_labels) if 'null' not in v})
    return pm


def read_pm_sol_kolmogorov(file):
    pm = dict()
    try:
        with open(file) as f:
            for line in f:
                if "-1" in line:
                    continue
                id_l, id_r = (int(x.strip()) for x in line.split(" "))
                pm.update({id_l: id_r})
    except FileNotFoundError:
        raise FileNotFoundError
    return pm


def read_seghyp_names(file):
    """Reads seghyp names from .ano.curated.aligned.txt file. Returns ordered list"""
    seghyp_names = []
    with open(file) as f:
        for line in f:
            parts = line.split(' ')
            seghyp_name = parts[1].strip().upper()
            seghyp_names.append(seghyp_name)
    return seghyp_names