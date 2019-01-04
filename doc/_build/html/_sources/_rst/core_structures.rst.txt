
Overview
===============

*pySpikeAnalysis* takes as input the Spike Sorting results and allow their analysis.
Currently only results from `SpyKING CIRCUS <https://spyking-circus.readthedocs.io/en/latest/>`_ are handled.

The Neo package
###############

The goal of `Neo <https://neo.readthedocs.io/en/0.6/index.html>`_ is to improve interoperability between Python tools for
analyzing, visualizing and generating electrophysiology data, by providing a common, shared object model.
In order to be as lightweight a dependency as possible, Neo is deliberately limited
to represention of data, with no functions for data analysis or visualization.

The Neo data model is the following :

.. image:: ./../_static/images/neo_base_schematic.png
   :scale: 80 %


NeoAll and Neo Epoch
####################

*pySpikeAnalysis* works with two main classes based on Neo : NeoAll and NeoEpoch.

NeoAll represents the spike sorting results from an entire file.
NeoEpoch allows to create epochs based on events such as stimuli onset, epileptic events, ...

While NeoAll allows to have a global view of the spiking activty across the whole file, NeoEpoch is useful
for analysis the behaviour of the different units in regard to certain events.

.. image:: ./../_static/images/neoall.png
   :scale: 60 %


.. image:: ./../_static/images/neoepoch.png
   :scale: 60 %

Possible analyses
#################

*pySpikeAnalysis* allows to run different analyses, summarized in the next image :

.. image:: ./../_static/images/outputs.png
   :scale: 50 %



