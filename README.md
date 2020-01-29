Restrained ensemble molecular dynamics method implementation for small angle scattering data. 
This is a custom force plugin for [OpenMM](http://openmm.org/) simulation package. Currently we provide only CUDA implementation of the plugin, so you’ll need CUDA-enebled device 
to be able to run the simulation. Besides current installation of OpenMM you’ll need [NumPy](https://numpy.org/) and [mpi4py](https://mpi4py.readthedocs.io), which is a python wrappers for MPI.

Building The Plugin
===================

This project uses [CMake](http://www.cmake.org) for its build system.  To build it, follow these
steps:

1. Create a directory in which to build the plugin.

2. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".

4. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.

5. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually,
this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.

6. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly
and that EXAMPLE_BUILD_CUDA_LIB is selected.

7. Press "Configure" again if necessary, then press "Generate".

8. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install`.

9. To run all the test cases build the "test" target, for example by typing `make test`.


Python API
==========

OpenMM uses [SWIG](http://www.swig.org) to generate its Python API.  SWIG takes an "interface
file", which is essentially a C++ header file with some extra annotations added, as its input.
It then generates a Python extension module exposing the C++ API in Python.

When building OpenMM's Python API, the interface file is generated automatically from the C++
API.  That guarantees the C++ and Python APIs are always synchronized with each other and avoids
the potential bugs that would come from have duplicate definitions.  It takes a lot of complex
processing to do that, though, and for a single plugin it's far simpler to just write the
interface file by hand.  You will find it in the "python" directory.

To build and install the Python API, build the "PythonInstall" target, for example by typing
"make PythonInstall".  (If you are installing into the system Python, you may need to use sudo.)
This runs SWIG to generate the C++ and Python files for the extension module
(REPluginWrapper.cpp and REplugin.py), then runs a setup.py script to build and
install the module.  Once you do that, you can use the plugin from your Python scripts:

    from simtk.openmm import System
    from REplugin import REForce
    system = System()
    force = REForce()
    system.addForce(force)

You might also need to change the variables in python/setup.py to your local path to OpenMM and the plugin directory

