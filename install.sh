#!/bin/bash

#########
# NOTES #
#########
# This script should get you "up-and-running" with:
#   . install.sh <virtualenv_directory_path>

# The assumption is that you have installed the base requirements (in `REQUIREMENTS` file) system-wide; these dependencies (e.g., numpy) are expected to be available system-wide, though they could optionally be installed in the virtual environment.

VENV_DIR=$1

echo "Creating the Python virtual environment..."
virtualenv -p /usr/bin/python3.5 --system-site-packages $VENV_DIR

echo "Installing and configuring system dependencies..."
# libxml2 and libxslt-dev (and lxml module) required for PyKML
sudo apt-get install libxml2 libxslt-dev python3-numpy python3-scipy python3-matplotlib python3-lxml

# The following are required to install scikit-learn with pip...
# See http://scikit-learn.org/stable/install.html
sudo apt-get install build-essential python3-dev python3-setuptools libopenblas-dev libblas-dev liblapack-dev libatlas-dev libatlas3gf-base
sudo update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3
sudo update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3
sudo apt-get install python3-setuptools
sudo easy_install3 pip

echo "Entering the virtual environment..."
source $VENV_DIR/bin/activate

# To fix problems with iPython shell...
# ipython3 # Run iPython, then exit
# hash -r

# To fix problems with matplotlib and pylab (ipython3)...
# rm ~/.cache/matplotlib/fontList.*.cache

# Install command-line tools for getting raster extents
mkdir ./unmixing/lib
git clone https://github.com/arthur-e/gdal_extent.py.git ./unmixing/lib/gdal_extent.py
chmod +x ./unmixing/lib/gdal_extent.py/gdal_extent.py
