from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

# Change the following variables to your local path to OpenMM and the plugin directory
openmm_dir = '@OPENMM_DIR@'
REplugin_header_dir = '@REPLUGIN_HEADER_DIR@'
REplugin_library_dir = '@REPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_REplugin',
                      sources=['REPluginWrapper.cpp'],
                      libraries=['OpenMM', 'REPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), REplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), REplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='REplugin',
      version='1.0',
      py_modules=['REplugin'],
      ext_modules=[extension],
     )
