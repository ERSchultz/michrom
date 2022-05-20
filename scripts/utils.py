import numpy as np
from sklearn.cluster import KMeans
import time
import os.path as osp
from seq2contact import triu_to_full, lammps_load, xyz_load, save_sc_contacts

def load_helper(args, contacts = False):
    xyz_file = osp.join(args.dir, 'data_out/output.xyz')
    lammps_file = osp.join(args.dir, 'traj.dump.lammpstrj')
    if osp.exists(xyz_file):
        xyz = xyz_load(xyz_file,
                    multiple_timesteps = True, save = True, N_min = 10,
                    down_sampling = args.down_sampling)
        x = None
    elif osp.exists(lammps_file):
        xyz, x = lammps_load(lammps_file, save = False, N_min = args.N_min,
                        down_sampling = args.down_sampling)

    if contacts:
        args.sc_contacts_dir = osp.join(args.odir, 'sc_contacts')
        save_sc_contacts(xyz, args.sc_contacts_dir, args.jobs, sparsify = True,
                        overwrite = True)

    return xyz, x

def contact_ratio(y_file = None):
    assert osp.exists(y_file), f'y_file does not exists: {y_file}'
    y = np.load(y_file)
    if len(y.shape) == 1:
        y = triu_to_full(y)

    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(y)
    m = len(y)
    seq = np.zeros((m, 2))
    seq[np.arange(m), kmeans.labels_] = 1

    a = seq[:, 0]
    b = seq[:, 1]
    result = R_a(y, a, b) * R_a(y, b, a)
    return result

def R_a(y, a, b):
    num = a @ y @ b
    denom = a @ y @ a
    result = num / denom
    return result

def rg(xyz):
    assert len(xyz.shape) == 2
    centroid = np.mean(xyz, axis = 0)
    return np.linalg.norm(xyz - centroid)

def end_to_end_distance(xyz):
    assert len(xyz.shape) == 2
    return np.linalg.norm(xyz[0] - xyz[-1])

def test():
    dir = '/home/erschultz/dataset_test3/samples/sample1'
    dir = '/home/erschultz/sequences_to_contact_maps/dataset_10_27_21/samples/sample40'
    y = np.load(osp.join(dir, 'y.npy'))
    a = contact_ratio(y)
    print(a)


    # t0 = time.time()
    # for _ in range(1000):
    #     xyz = np.random.rand(300, 3)
    #     a = rg(xyz)
    # tf = time.time()
    # print(tf - t0)





if __name__ == '__main__':
    test()
