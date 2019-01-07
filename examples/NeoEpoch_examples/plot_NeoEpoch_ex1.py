"""
=====================
Neo Epoch - example 1 - Rasterplots
=====================

This example shows how to use NeoEpoch

"""

from neoStructures import *
import matplotlib.pyplot as plt
from os.path import isdir, join, isfile
import pickle
import seaborn as sns
sns.set()
sns.set_context('paper')

import sys
print(sys.version)

#######################
# Load neoEpoch data with _pickle
data_dir = join('pySpikeAnalysis', 'sample_data') if isdir('pySpikeAnalysis') else join('..', '..', 'pySpikeAnalysis', 'sample_data')
print(data_dir)
neo_epoch_filename = r'neoepoch_070119_1117.p'
if not isfile(join(data_dir, neo_epoch_filename)):
    raise ValueError('Cannot find the file {}'.format(join(data_dir, neo_epoch_filename)))
with open(join(data_dir, neo_epoch_filename), 'rb') as f:
    neo_epoch = pickle.load(f)

neo_epoch.save_fig = 0

##############################
# See information about NeoAll
print(neo_epoch)

##############################
# Plot the raster plot for unit 4
neo_epoch.plot_rasterplot(4)

