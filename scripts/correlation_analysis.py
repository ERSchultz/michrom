import argparse
import multiprocessing
import os
import os.path as osp

import numpy as np
from seq2contact import (ArgparserConverter, lammps_load, pearson_round,
                         plot_xyz_gif, xyz_load)
from sklearn.linear_model import LinearRegression
from utils import (MomentofInertia, contact_ratio, end_to_end_distance,
                   load_helper, rg)


def getArgs(default_dir='/home/erschultz/dataset_test/samples/sample1'):
    parser = argparse.ArgumentParser(description='Base parser')
    AC = ArgparserConverter()

    parser.add_argument('--dir', type=str, default=default_dir,
                        help='location of data')
    parser.add_argument('--odir', type=str,
                        help='location to write to')
    parser.add_argument('--N_min', type=int, default=2000,
                        help='minimum sample index to keep')
    parser.add_argument('--mode', type=str, default='contact_diffusion')
    parser.add_argument('--update_mode', type=str, default='kNN')
    parser.add_argument('--k', type=int, default=4,
                        help='k for update_mode')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--sparse_format', action='store_true',
                        help='True to store sc_contacts in sparse format')
    parser.add_argument('--down_sampling', type=int, default=10)

    args = parser.parse_args()
    if args.odir is None:
        args.odir = args.dir
    elif not osp.exists(args.odir):
        os.mkdir(args.odir, mode = 0o755)

    fname = f'{args.mode}_{args.update_mode}{args.k}'
    args.odir_mode = osp.join(args.odir, fname)

    print(args)
    return args

def correlation_analysis(args, xyz = None, x = None):
    args.log_file_path = osp.join(args.odir_mode, 'out.log')
    args.log_file = open(args.log_file_path, 'a')

    if xyz is None:
        xyz, x = load_helper(args)
    N, _, _ = xyz.shape

    corr_stats = CorrelationStats(xyz, args.log_file)

    sc_contacts_dir = osp.join(args.odir_mode, 'iteration_0/sc_contacts')

    # find max it
    max_it = np.max([int(f[10:]) for f in os.listdir(args.odir_mode) if f.startswith('iteration')])
    N = max_it + 1
    print(f'max_it: {max_it}')
    print('\nIteration 0', file = args.log_file)
    corr_stats.correlate_contact_ratio(sc_contacts_dir, mode = 'spearman')
    corr_stats.correlate_rg(mode = 'spearman')
    corr_stats.correlate_end_to_end_distance(mode = 'spearman')
    corr_stats.correlate_moi(mode='spearman')
    plot_xyz_gif(xyz, x, args.odir_mode, ofile = 'xyz_time.gif')


    for it in range(1, max_it+1):
        print(f'\nIteration {it}', file = args.log_file)
        print(f'\nIteration {it}')
        it_dir = osp.join(args.odir_mode, f'iteration_{it}')
        sc_contacts_dir = osp.join(it_dir, 'sc_contacts')
        if not osp.exists(sc_contacts_dir):
            # check if iteration failed
            continue

        # time
        corr_stats.correlate_contact_ratio(sc_contacts_dir, mode = 'spearman')

        # eigenspace
        v_file = osp.join(it_dir, 'v.npy')
        if osp.exists(v_file):
            v = np.load(v_file)
            corr_stats.correlate_contact_ratio(sc_contacts_dir, v)
            corr_stats.correlate_rg(v)
            corr_stats.correlate_end_to_end_distance(v)
            corr_stats.correlate_moi(v)

        # traj
        order = np.loadtxt(osp.join(it_dir, 'order.txt'), dtype = np.int32)
        plot_xyz_gif(xyz, x, args.odir_mode, ofile = f'xyz_traj{it}.gif', order = order)



class CorrelationStats():
    def __init__(self, xyz, log_file):
        self.xyz = xyz
        assert len(xyz.shape) == 3, f'invalid shape for xyz: {xyz.shape}'
        self.N, _, _ = xyz.shape

        self.log_file = log_file

        self.end_to_end_vals = None
        self.rg_vals = None
        self.principal_moi = None

    def correlate_contact_ratio(self, dir, v = None, mode = 'both'):
        vals = np.zeros((self.N), dtype=np.float64)
        for i in range(self.N):
            vals[i] = contact_ratio(osp.join(dir, f'y_sc_{i}.npy'))

        if v is None:
            self._correlate_inner(vals, np.arange(self.N), mode, 'Time vs Rg')
        else:
            for i in range(1, 4): # eig 0 is trivial
                v_i = v[:, i]
                self._correlate_inner(vals, v_i, mode, f'v_{i} vs contact_ratio')
            self._linear_combination_inner(v, vals)

    def correlate_rg(self, v = None, mode = 'both'):
        if self.rg_vals is None:
            self.rg_vals = np.zeros((self.N), dtype=np.float64)
            for i in range(self.N):
                self.rg_vals[i] = rg(self.xyz[i])

        if v is None:
            self._correlate_inner(self.rg_vals, np.arange(self.N), mode, 'Time vs Rg')
        else:
            for i in range(1, 4): # eig 0 is trivial
                v_i = v[:, i]
                self._correlate_inner(self.rg_vals, v_i, mode, f'v_{i} vs Rg')
            self._linear_combination_inner(v, self.rg_vals)

    def correlate_end_to_end_distance(self, v = None, mode = 'both'):
        if self.end_to_end_vals is None:
            self.end_to_end_vals = np.zeros((self.N), dtype=np.float64)
            for i in range(self.N):
                self.end_to_end_vals[i] = end_to_end_distance(self.xyz[i])

        if v is None:
            self._correlate_inner(self.end_to_end_vals, np.arange(self.N), mode, 'Time vs end_to_end_distance')
        else:
            for i in range(1, 4): # eig 0 is trivial
                v_i = v[:, i]
                self._correlate_inner(self.end_to_end_vals, v_i, mode, f'v_{i} vs end_to_end_distance')
            self._linear_combination_inner(v, self.end_to_end_vals)


    def correlate_moi(self, v = None, mode = 'both'):
        if self.principal_moi is None:
            self.principal_moi = np.zeros((self.N, 3), dtype=np.float64)
            for i in range(self.N):
                self.principal_moi[i] = MomentofInertia.get_principal_moi(self.xyz[i])

        if v is None:
            v = np.arange(self.N)
            self._correlate_inner(self.principal_moi[:, 0], v, mode, f'Time vs I_a')
            self._correlate_inner(self.principal_moi[:, 1], v, mode, f'Time vs I_b')
            self._correlate_inner(self.principal_moi[:, 2], v, mode, f'Time vs I_c')
        else:
            for i in range(1, 4): # eig 0 is trivial
                v_i = v[:, i]
                self._correlate_inner(self.principal_moi[:, 0], v_i, mode, f'v_{i} vs I_a')
                self._correlate_inner(self.principal_moi[:, 1], v_i, mode, f'v_{i} vs I_b')
                self._correlate_inner(self.principal_moi[:, 2], v_i, mode, f'v_{i} vs I_c')
            self._linear_combination_inner(v, self.principal_moi)

    def _correlate_inner(self, a, b, mode, label):
        if mode == 'both':
            pearson = pearson_round(a, b, stat = 'pearson')
            spearman = pearson_round(a, b, stat = 'spearman')
            print(f'{label}: pearson={pearson}, spearman={spearman}',
                    file = self.log_file)
        elif mode == 'pearson':
            pearson = pearson_round(a, b, stat = 'pearson')
            spearman = None
            print(f'{label}: pearson={pearson}', file = self.log_file)
        elif mode == 'spearman':
            pearson = None
            spearman = pearson_round(a, b, stat = 'spearman')
            print(f'{label}: spearman={spearman}', file = self.log_file)
        else:
            raise Exception(f'Invalid mode: {mode}')

    def _linear_combination_inner(self, x, y):
        reg = LinearRegression()
        reg.fit(x, y)
        score = reg.score(x, y)
        print(f'Linear Regression R^2: {score}', file = self.log_file)

def main():
    args = getArgs()
    correlation_analysis(args)

def test():
    args = getArgs()
    xyz, x = load_helper(args)
    N, _, _ = xyz.shape
    corr_stats = CorrelationStats(xyz)
    sc_contacts_dir = osp.join(args.odir_mode, 'iteration_0/sc_contacts')

    it = 1
    it_dir = osp.join(args.odir_mode, f'iteration_{it}')
    sc_contacts_dir = osp.join(it_dir, 'sc_contacts')


    # eigenspace
    v_file = osp.join(it_dir, 'v.npy')
    v = np.load(v_file)
    v_1 = v[:20, 1]
    order = np.argsort(v_1)
    spearman = pearson_round(v_1, order, stat = 'spearman')
    print(v_1)
    print(order)
    print(v_1[order])
    print(np.arange(N)[order])
    print(spearman)

if __name__ == '__main__':
    main()
    # test()
