"""elastic augmentation based on augment-nd package of funkey"""
import random

import augment


def create_elastic_transformation(shape, control_point_spacing,
                                  jitter_sigma, rotation_interval,
                                  subsample):
    transformation = augment.create_identity_transformation(shape, subsample)
    transformation += augment.create_elastic_transformation(shape,
                                                            control_point_spacing,
                                                            jitter_sigma, subsample)
    rotation = random.random() * (rotation_interval[1] - rotation_interval[
        0]) + rotation_interval[0]
    if rotation != 0:
        transformation += augment.create_rotation_transformation(shape,
                                                                 rotation, subsample)
    if subsample > 1:
        transformation = augment.upscale_transformation(transformation, shape)
    return transformation
