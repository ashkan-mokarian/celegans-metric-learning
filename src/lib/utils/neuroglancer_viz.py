import neuroglancer


def neuroglancer_init():
    neuroglancer.set_server_bind_address('127.0.0.1', '12345')
    viewer = neuroglancer.Viewer()
    dimensions = neuroglancer.CoordinateSpace(
        names=['x', 'y', 'z'],
        units='',
        scales=[1, 1, 1])
    with viewer.txn() as s:
        s.dimensions = dimensions
    return viewer


def neuroglancer_addlayer(txn, name, data, volume_type, offset=[0, 0, 0]):
    txn.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=data,
                volume_type=volume_type,
                dimensions=neuroglancer.CoordinateSpace(
                    names=['x', 'y', 'z'],
                    units=['', '', ''],
                    scales=[1, 1, 1],
                    coordinate_arrays=[
                    None, None, None
                ]),
            voxel_offset=offset
    ))