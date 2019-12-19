.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3585979.svg
   :target: https://doi.org/10.5281/zenodo.3585979

==========================
Interactive Unmixing Tools
==========================

Overview
========

This is a library of interactive tools and functions for performing linear spectral mixture analysis (LSMA) and spatially adaptive spectral mixture analysis (SASMA).
It supports parallel fully constrained least-squares (FCLS) mixture analysis over multiple processes, allowing for very efficient mapping of endmember abundances, both in the spatially adaptive approach and in regular LSMA.
In detail, the `unmixing` module includes tools and functions to support:

- Stacking raster bands from any GDAL file format, including HDF;
- Applying a mask or a combination of masks to a raster;
- Generating raster masks from saturated pixels or using density slicing (e.g., to mask water pixels);
- Applying CFMask output and other quality assurance layers from the USGS;
- Band-wise compositing of reflectance (or fraction/ abundance images);
- Tasseled cap transformation and other indices including NDVI and RNDSI;
- Dimensionality reduction through the Minimum Noise Fraction (MNF);
- Radiometric rectification of a raster time series;
- Visualizing the mixing space of a moderate resolution raster;
- Interactive selection of pixels in the mixing space, which are then captured in a KML file;
- Endmember induction and plotting of endmembers for linear spectral mixture analysis (LSMA);
- Fully constrained least squares (FCLS) unmixing;
- Spatial interpolation of endmembers for spatially adaptive spectral mixture analysis (SASMA);
- Learning of endmember candidates for SASMA using classification and regression trees (CART);
- Validation of unmixing through a forward model of reflectance;

**For an overview and tutorial on how to use this library, check out the iPython Notebook(s) in the** ``docs/`` **folder:**

- <https://github.com/arthur-e/unmixing/blob/master/docs/Overview_LSMA_in_Python.ipynb>
- <https://github.com/arthur-e/unmixing/blob/master/docs/Example_Spatially_Adaptive_Spectral_Mixture_Analysis_SASMA.ipynb>

Installation and Setup
======================

Because this is a scientific library, there are complex dependencies that may be difficult to install.
For GNU/Linux systems, particularly Ubuntu, look at ``install.sh`` for a guide on installing the system dependencies required for the Python dependencies.

At this time, installation is intended for development purposes only.
As such, ``unmixing`` should be installed in "editable" mode using ``pip``; see <https://packaging.python.org/distributing/#working-in-development-mode>.
From the ``unmixing`` directory, where ``setup.py`` is found::

    $ pip install -e .

Dependencies
------------

* ``numpy``
* ``scipy``
* ``matplotlib``
* ``pysptools``
* ``GDAL``
* ``pykml``
* ``scikit-learn`` (For SASMA)

Use System Dependencies
-----------------------

Many of the core packages, particularly NumPy and SciPy, have wide adoption and use.
They also take a lot of time and clock cycles to compile into a virtual environment.
Consequently, it is recommended that these libraries be installed globally (system-wide).

Other packages should be installed only in the virtual environment.
To use both system-wide and local packages within a virtual environment, the virtual environment must be set up with the ``--system-site-packages`` option::

    sudo apt-get install python3-numpy python3-scipy python3-matplotlib python3-zmq
    virtualenv -p /usr/bin/python3.5 --system-site-packages <virtualenv_path>

The local packages can be installed within the virtual environment via ``pip``::

    source /my/virtualenv/bin/activate
    pip install -r REQUIREMENTS


Documentation
=============

Python Dependencies
-------------------

Some resources on the dependencies can be found:

* ``matplotlib`` <http://matplotlib.org/contents.html>
* PySptools <http://pysptools.sourceforge.net/index.html>
* PySptools on PyPI <https://pypi.python.org/pypi/pysptools/0.13.0>

Third-Party Tools
-----------------

Documentation on related (but not required) third-party tools can be found:

* Fmask <https://code.google.com/p/fmask/>
* CFmask <https://code.google.com/p/cfmask/> (A C version of Fmask)
* LEDAPS <https://code.google.com/p/ledaps/>

Foundations
===========

Hyperspectral Imaging (HSI) Cubes
---------------------------------

Many of the tools in this library are designed to work with HSI cubes.
While the raster arrays read in from GDAL are ``p x m x n`` arrays, HSI cubes are ``n x m x p`` arrays, where ``m`` is the row index, `n` is the column index, and ``p`` is the spectral band number.
It is important to note that the row index `n` corresponds to the latitude while the column index `m` corresponds to the longitude.
Thus, the coordinates ``(n, m)`` can be converted directly to a longitude-latitude pair.
