# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yevhen/projects/openmm_plugin_saxs/source

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yevhen/projects/openmm_plugin_saxs/source

# Utility rule file for PythonInstall.

# Include the progress variables for this target.
include python/CMakeFiles/PythonInstall.dir/progress.make

python/CMakeFiles/PythonInstall: python/ExamplePluginWrapper.cpp


python/ExamplePluginWrapper.cpp: python/exampleplugin.i
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/yevhen/projects/openmm_plugin_saxs/source/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ExamplePluginWrapper.cpp"
	cd /home/yevhen/projects/openmm_plugin_saxs/source/python && /usr/local/bin/swig -python -c++ -o ExamplePluginWrapper.cpp -I/home/yevhen/programs/openmm/include /home/yevhen/projects/openmm_plugin_saxs/source/python/exampleplugin.i

PythonInstall: python/CMakeFiles/PythonInstall
PythonInstall: python/ExamplePluginWrapper.cpp
PythonInstall: python/CMakeFiles/PythonInstall.dir/build.make
	cd /home/yevhen/projects/openmm_plugin_saxs/source/python && /usr/bin/python setup.py build
	cd /home/yevhen/projects/openmm_plugin_saxs/source/python && /usr/bin/python setup.py install
.PHONY : PythonInstall

# Rule to build all files generated by this target.
python/CMakeFiles/PythonInstall.dir/build: PythonInstall

.PHONY : python/CMakeFiles/PythonInstall.dir/build

python/CMakeFiles/PythonInstall.dir/clean:
	cd /home/yevhen/projects/openmm_plugin_saxs/source/python && $(CMAKE_COMMAND) -P CMakeFiles/PythonInstall.dir/cmake_clean.cmake
.PHONY : python/CMakeFiles/PythonInstall.dir/clean

python/CMakeFiles/PythonInstall.dir/depend:
	cd /home/yevhen/projects/openmm_plugin_saxs/source && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yevhen/projects/openmm_plugin_saxs/source /home/yevhen/projects/openmm_plugin_saxs/source/python /home/yevhen/projects/openmm_plugin_saxs/source /home/yevhen/projects/openmm_plugin_saxs/source/python /home/yevhen/projects/openmm_plugin_saxs/source/python/CMakeFiles/PythonInstall.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : python/CMakeFiles/PythonInstall.dir/depend

