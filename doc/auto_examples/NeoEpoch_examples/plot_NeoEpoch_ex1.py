"""
=====================
Neo Epoch - example 1 - Rasterplots
=====================

This example shows how to use NeoEpoch

"""

from neoStructures import *
import matplotlib.pyplot as plt
from os.path import isdir, join
import _pickle
import seaborn as sns
sns.set()
sns.set_context('paper')

import sys
print(sys.version)

#######################
# Load neoEpoch data with _pickle
data_dir = join('pySpikeAnalysis', 'sample_data') if isdir('pySpikeAnalysis') else join('..', '..', 'pySpikeAnalysis', 'sample_data')
neo_epoch_filename = r'neoepoch_071118_1132.p'
with open(join(data_dir, neo_epoch_filename), 'rb') as f:
    neo_epoch = _pickle.load(f)

neo_epoch.save_fig = 0

##############################
# See information about NeoAll
print(neo_epoch)

##############################
# Plot the raster plot for unit 4
neo_epoch.plot_rasterplot(4)

