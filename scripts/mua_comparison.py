# Compare supposed multi unit by displaying the template for each pair of supposed MUA
from neoStructures import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
sns.set()
sns.set_context('paper')

spykingcircus_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\spykingcircus_results'
results_filename = r'20170113-104634-001_0'
event_filepath = r''
signal_dirpath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\data\edf_30kHz'
probe_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\014_VL.prb'
artefact_csv_filepath = r'C:\Users\deudon\Desktop\SpikeSorting\_Data\VL14\DAY2_Elodie\artefact_day2_epifar.csv'

neoAll = NeoAll(spykingcircus_dirpath, results_filename, probe_filepath=probe_filepath, signal_dirpath=signal_dirpath,
                event_filepath=[], fig_dirpath=[], save_fig=0)

print(neoAll)

mua_pos = np.array([1, 2, 3, 5, 6, 9, 11])
n_mua = mua_pos.size
ymin, ymax = 0,0
f = plt.figure()
for i in range(n_mua):
    ax = plt.subplot2grid((1, n_mua), (0, i))
    neoAll.plot_unit_shape(unit_pos=mua_pos[i], ax=ax)
    ylims = ax.get_ylim()
    if ylims[0] < ymin:
        ymin = ylims[0]
    if ylims[1] > ymax:
        ymax = ylims[1]


f = plt.figure()
for i in range(n_mua):
    for j in range(n_mua):
        if j >= i:
            continue
        ax_i = plt.subplot2grid((n_mua, 2*n_mua), (i, 2*j+1))
        neoAll.plot_unit_shape(unit_pos=mua_pos[i], ax=ax_i)
        ax_i.get_legend().remove()
        ax_i.set_title('Unit {}'.format(mua_pos[i]))
        ax_i.set(xlabel='', ylabel='')
        ax_i.set_xticks([])
        # ax_i.set_yticks([])
        ax_i.set_ylim((ymin, ymax))
        ax_j = plt.subplot2grid((n_mua, 2*n_mua), (i, 2*j))
        neoAll.plot_unit_shape(unit_pos=mua_pos[j], ax=ax_j)
        ax_j.get_legend().remove()
        ax_j.set_title('Unit {}'.format(mua_pos[j]))
        ax_j.set(xlabel='', ylabel='')
        ax_j.set_xticks([])
        ax_j.set_ylim((ymin, ymax))
        # ax_j.set_yticks([])


def compare_2_units(neoAll, unit_i, unit_j):
    f = plt.figure()
    ax_i = f.add_subplot(121)
    neoAll.plot_unit_shape(unit_pos=unit_i, ax=ax_i)
    ylim_i = ax_i.get_ylim()
    ax_j = f.add_subplot(122)
    neoAll.plot_unit_shape(unit_pos=unit_j, ax=ax_j)
    ylim_j = ax_j.get_ylim()
    ax_i.set_ylim(min(ylim_i[0],ylim_j[0]), max(ylim_i[1], ylim_j[1]))
    ax_j.set_ylim(min(ylim_i[0],ylim_j[0]), max(ylim_i[1], ylim_j[1]))


compare_2_units(neoAll, 1, 11)