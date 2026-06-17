cmake_minimum_required(VERSION 3.15)
project(maeher CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# ── REAPER upstream sources ──────────────────────────────────────────────────
set(REAPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp/reaper")

set(REAPER_SOURCES
    ${REAPER_DIR}/core/file_resource.cc
    ${REAPER_DIR}/core/float_matrix.cc
    ${REAPER_DIR}/core/track.cc
    ${REAPER_DIR}/epoch_tracker/epoch_tracker.cc
    ${REAPER_DIR}/epoch_tracker/fd_filter.cc
    ${REAPER_DIR}/epoch_tracker/fft.cc
    ${REAPER_DIR}/epoch_tracker/lpc_analyzer.cc
)

# ── Python extension ─────────────────────────────────────────────────────────
pybind11_add_module(_core
    cpp/bindings.cpp
    ${REAPER_SOURCES}
)

target_include_directories(_core PRIVATE
    ${REAPER_DIR}/..          # so #include "reaper/..." resolves
    ${REAPER_DIR}             # so upstream headers can #include "core/..." etc.
)

# Silence warnings from upstream code we don't own.
target_compile_options(_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wno-unused-variable
        -Wno-sign-compare
        -Wno-reorder
    >
)

install(TARGETS _core DESTINATION maeher)
