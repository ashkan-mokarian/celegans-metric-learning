"""Functionalities regarding the combinatorial pairwise matching solutions"""
import logging
import os

from lib.data.labels import Labels as uLabels

logger = logging.getLogger(__name__)


def read_pm_sol(file):
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


def read_nuclei_names(file):
    nuclei_names = list()
    try:
        with open(file) as f:
            for line in f:
                nuclei_names.append(line.strip().upper())
    except FileNotFoundError:
        raise FileNotFoundError
    return nuclei_names


def get_pm_ulabels(pm_sol, nuclei_names1, nuclei_names2, ulabels):
    if not type(pm_sol) is dict:
        if not os.path.exists(pm_sol):
            logger.error('pm_sol parameter has to be a pm solution(dict) or the path to a solution file(file_path)')
            raise ValueError
        pm_sol = read_pm_sol(pm_sol)

    if not type(nuclei_names1) is list:
        if not os.path.exists(nuclei_names1):
            logger.error('nuclei_names1 parameter has to be nuclei names(list) or the path to the corresponding file('
                         'file_path')
            raise ValueError
        nuclei_names1 = read_nuclei_names(nuclei_names1)

    if not type(nuclei_names2) is list:
        if not os.path.exists(nuclei_names2):
            logger.error('nuclei_names1 parameter has to be nuclei names(list) or the path to the corresponding file('
                         'file_path')
            raise ValueError
        nuclei_names2 = read_nuclei_names(nuclei_names2)

    if not type(ulabels) is uLabels:
        if not os.path.exists(ulabels):
            logger.error('ulabels parameter has to be unique labels variable(Labels) or the path to the universe '
                         'file(file_path')
            raise ValueError
        ulabels = ulabels(ulabels)

    # read all the pairwise matching key and values and find the corresponding uid in the uLabels file and replace it,
    # if does not exist, then omit it as it is counted as not_valid
    pm_u = dict()
    for lid,rid in pm_sol.items():

        lname = nuclei_names1[lid].upper()
        rname = nuclei_names2[rid].upper()

        if ulabels.is_valid_label(lname) and ulabels.is_valid_label(rname):
            pm_u.update({ulabels.label_to_uid(lname): ulabels.label_to_uid(rname)})

    return pm_u