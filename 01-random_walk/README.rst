.. Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

###########
Random Walk
###########

.. default-role:: math

This is the work on the Computational Physics exercise by Martin Ueding.

.. Please see the responding HTML for a nicer version of this document.

I implemented this using C and CUDA. Random numbers are generated with the
``curand`` library. To compile and execute this, please run::

    cmake .
    make
    ./random-walk

In the ``CMakeLists.txt`` I have fixed the CUDA target to 1.1 since I have a
GeForce 9500 GT at home. This has to be adjusted for other GPUs, I believe.

To generate the plots, run ``python3 analysis.py``. This requires ``numpy`` and
``matplotlib``. The output files are included in this archive.

The program generates ``averages.csv`` and ``rms.csv`` which contain the
average distance and the RMS of the distance `d` versus the number of steps
`n`. In the loglog plot one can see that the expected relationship `d \propto
\sqrt n` holds:

.. figure:: plot.png

.. vim: spell tw=79
