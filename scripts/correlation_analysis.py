import numpy as np
import os.path as osp
from seq2contact import pearson_round
import os
import argparse
import multiprocessing

from seq2contact import ArgparserConverter, xyz_load, lammps_load, plot_xyz_gif
from utils import contact_ratio, end_to_end_distance, rg, load_helper

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

    corr_stats = CorrelationStats(xyz)

    sc_contacts_dir = osp.join(args.odir_mode, 'iteration_0/sc_contacts')

    # find max it
    max_it = np.max([int(f[10:]) for f in os.listdir(args.odir_mode) if f.startswith('iteration')])
    N = max_it + 1
    print(f'max_it: {max_it}')


    print('\nIteration 0', file = args.log_file)
    _, spearman = corr_stats.correlate_contact_ratio(sc_contacts_dir, mode = 'spearman')
    print(f'Time vs contact_Ratio: spearman={spearman}',
            file = args.log_file)
    _, spearman = corr_stats.correlate_rg(mode = 'spearman')
    print(f'Time vs Rg: spearman={spearman}',
            file = args.log_file)
    _, spearman = corr_stats.correlate_end_to_end_distance(mode = 'spearman')
    print(f'Time vs end_to_end_distance: spearman={spearman}',
            file = args.log_file)
    plot_xyz_gif(xyz, x, args.odir_mode, ofile = 'xyz_time.gif')


    for it in range(1, max_it+1):
        print(f'\nIteration {it}', file = args.log_file)
        print(f'\nIteration {it}')
        it_dir = osp.join(args.odir_mode, f'iteration_{it}')
        sc_contacts_dir = osp.join(it_dir, 'sc_contacts')

        # time
        _, spearman = corr_stats.correlate_contact_ratio(sc_contacts_dir, mode = 'spearman')
        print(f'Time vs contact_Ratio: spearman={spearman}',
                file = args.log_file)

        # eigenspace
        v_file = osp.join(it_dir, 'v.npy')
        if osp.exists(v_file):
            v = np.load(v_file)
            for i in range(1, 2): # eig 0 is trivial
                v_i = v[:, i]
                pearson, spearman = corr_stats.correlate_contact_ratio(sc_contacts_dir, v_i)
                print(f'v_{i} vs contact_Ratio: pearson={pearson}, spearman={spearman}',
                        file = args.log_file)
                pearson, spearman = corr_stats.correlate_rg(v_i)
                print(f'v_{i} vs Rg: pearson={pearson}, spearman={spearman}',
                        file = args.log_file)
                pearson, spearman = corr_stats.correlate_end_to_end_distance(v_i)
                print(f'v_{i} vs end_to_end_distance: pearson={pearson}, spearman={spearman}',
                        file = args.log_file)


        # traj
        order = np.loadtxt(osp.join(it_dir, 'order.txt'), dtype = np.int32)
        plot_xyz_gif(xyz, x, args.odir_mode, ofile = f'xyz_traj{it}.gif', order = order)

class CorrelationStats():
    def __init__(self, xyz):
        self.xyz = xyz
        assert len(xyz.shape) == 3, f'invalid shape for xyz: {xyz.shape}'
        self.N, _, _ = xyz.shape

        self.end_to_end_vals = None
        self.rg_vals = None

    def correlate_contact_ratio(self, dir, v = None, mode = 'both', jobs = 1):
        if jobs == 1:
            vals = np.zeros((self.N), dtype=np.float64)
            for i in range(self.N):
                vals[i] = contact_ratio(osp.join(dir, f'y_sc_{i}.npy'))
        else:
            # this didnt' seem to help
            mapping = [osp.join(dir, f'y_sc_{i}.npy') for i in order]
            with multiprocessing.Pool(jobs) as p:
                vals = p.map(contact_ratio, mapping)

        if v is None:
            v = np.arange(self.N)

        return self._correlate_inner(vals, v, mode)

    def correlate_rg(self, v = None, mode = 'both'):
        if self.rg_vals is None:
            self.rg_vals = np.zeros((self.N), dtype=np.float64)
            for i in range(self.N):
                self.rg_vals[i] = rg(self.xyz[i])

        if v is None:
            v = np.arange(self.N)

        return self._correlate_inner(self.rg_vals, v, mode)

    def correlate_end_to_end_distance(self, v = None, mode = 'both'):
        if self.end_to_end_vals is None:
            self.end_to_end_vals = np.zeros((self.N), dtype=np.float64)
            for i in range(self.N):
                self.end_to_end_vals[i] = end_to_end_distance(self.xyz[i])

        if v is None:
            v = np.arange(self.N)


        return self._correlate_inner(self.end_to_end_vals, v, mode)

    def _correlate_inner(self, a, b, mode):
        if mode == 'both':
            pearson = pearson_round(a, b, stat = 'pearson')
            spearman = pearson_round(a, b, stat = 'spearman')
        elif mode == 'pearson':
            pearson = pearson_round(a, b, stat = 'pearson')
            spearman = None
        elif mode == 'spearman':
            pearson = None
            spearman = pearson_round(a, b, stat = 'spearman')

        return pearson, spearman

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
