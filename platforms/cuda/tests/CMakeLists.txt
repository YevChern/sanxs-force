#
# Testing
#

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIR})

# Automatically create tests using files named "Test*.cpp"
FILE(GLOB TEST_PROGS "*Test*.cpp")
FOREACH(TEST_PROG ${TEST_PROGS})
    GET_FILENAME_COMPONENT(TEST_ROOT ${TEST_PROG} NAME_WE)

    # Pteros
    SET(PTEROS_INCLUDE_PATH "/home/yevhen/programs/pteros/include")
    SET(PTEROS_LIB_PATH "/home/yevhen/programs/pteros/lib")
    IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -msse2 -std=c++11 -fopenmp")
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    ENDIF()


    find_package(Boost 1.48.0 REQUIRED system date_time)

    include_directories(${PTEROS_INCLUDE_PATH} ${Boost_INCLUDE_DIRS})
    link_directories(${PTEROS_LIB_PATH})

    # Link with shared library
    ADD_EXECUTABLE(${TEST_ROOT} ${TEST_PROG})
    TARGET_LINK_LIBRARIES(${TEST_ROOT} ${SHARED_RE_TARGET} ${SHARED_TARGET} pteros pteros_analysis ${Boost_LIBRARIES})
    IF (APPLE)
        SET_TARGET_PROPERTIES(${TEST_ROOT} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
    ELSE (APPLE)
        SET_TARGET_PROPERTIES(${TEST_ROOT} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
    ENDIF (APPLE)
    ADD_TEST(${TEST_ROOT}Single ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} single)
    ADD_TEST(${TEST_ROOT}Mixed ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} mixed)
    ADD_TEST(${TEST_ROOT}Double ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} double)

ENDFOREACH(TEST_PROG ${TEST_PROGS})
