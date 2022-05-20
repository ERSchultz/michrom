import sys

for p in ['/home/erschultz', '/home/erschultz/sequences_to_contact_maps']:
    sys.path.insert(1, p)

from sequences_to_contact_maps.utils.argparse_utils import (ArgparserConverter)
from sequences_to_contact_maps.utils.plotting_utils import (plot_matrix,
                                                            plot_seq_binary,
                                                            plot_seq_exclusive,
                                                            plot_top_PCs)
from sequences_to_contact_maps.utils.utils import (LETTERS,
                                                   DiagonalPreprocessing,
                                                   calc_dist_strat_corr, crop,
                                                   pearson_round)
from sequences_to_contact_maps.utils.xyz_utils import lammps_load, xyz_load
from sequences_to_contact_maps.utils.utils import triu_to_full
from sequences_to_contact_maps.utils.argparse_utils import ArgparserConverter
from sequences_to_contact_maps.utils.load_utils import save_sc_contacts
from sequences_to_contact_maps.utils.plotting_utils import plot_matrix, plot_sc_contact_maps_inner, plot_xyz_gif
from sequences_to_contact_maps.utils.utils import (DiagonalPreprocessing, pearson_round, print_size,
                         print_time, triu_to_full)
from sequences_to_contact_maps.utils.xyz_utils import (lammps_load, xyz_load, xyz_to_contact_grid)
