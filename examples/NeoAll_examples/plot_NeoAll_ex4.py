"""
===========================
    Neo All - example 4
===========================

In this example, we look at the single unit activity around interictal epileptic discharges (IEDs). To do this we will
add events and define epochs to the NeoAll instance.

"""

from neoStructures import *
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isdir, join
import seaborn as sns
import _pickle
sns.set()
sns.set_context('paper')

################################################
# In this example we will load the NeoAll instance from a pickle file
data_dir = join('pySpikeAnalysis', 'sample_data') if isdir('pySpikeAnalysis') else join('..', '..', 'pySpikeAnalysis', 'sample_data')
neo_all_filename = r'neoall_071118_1132.p'
with open(join(data_dir, neo_all_filename), 'rb') as f:
    neoAll = _pickle.load(f)

################################
# See information about NeoAll :
print(neoAll)

#####################################
# Before creating epochs, we first need to create events. They can represents stimuli onset or epileptic events for
# instance. In this example, each event correponds to an Interictal Epileptic Discharge. There are two ways to create
# events, either by providing the event_filepath parameter when instanciating a NeoAll instance or by calling the
# :func:`neoStructures.NeoAll.read_event_file` method. The indexing of the csv file (i.e. to which field correspond each
# column) is done in the **neoStructures_params.py** file.
# Here we can see that the neoAll already contains events :
print(neoAll.segments[0].events)

###############################
# We can access the times of the events and the name of the events :
print('Event name : {}'.format(neoAll.segments[0].events[0].name))
print('Event times : {}'.format(neoAll.segments[0].events[0].times))

##################################
# Let's now define epochs on each segments (NeoAll class contains one Neo.segment for each unit), using the
# :func:`neoStructures.NeoAll.create_epochs_around_events`
# We have to specify the epoch onset and duration (s), as well as the epochs name.
epoch_t_start = np.array([-0.500, -0.250, -0.050, 0.050, 0.250])
epoch_duration = np.array([0.250, 0.200, 0.100, 0.200, 0.250])
epoch_names = ['Pre-IED baseline', 'Pre-IED', 'IED', 'Slow wave', 'Post-IED']

##################################
# We must specify around which event we want to create epochs
neoAll.create_epochs_around_events('Pointe', time_offset=epoch_t_start, epoch_duration=epoch_duration,
                                   epoch_names=epoch_names)

##################################
# We can now study the behaviour of neurons on the different periods
# of the IEDs we have defined.
neoAll.plot_spikerate_change_ratio_on_epochs('IED', 'Pre-IED')

###################################
# We can see that some units start firing during the IED and are almost silent before, in the Pre-IED period. Further
# analyses can be conducted using the NeoEpoch class which allows to plot rasterplot of the different units around
# the events of interest.

