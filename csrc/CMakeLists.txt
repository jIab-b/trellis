cmake_minimum_required(VERSION 3.18)

project(trellis_infer LANGUAGES CXX)

option(TRELLIS_WITH_CUDA "Build with CUDA support" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect CUDA optionally
if(TRELLIS_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
    set(HAVE_CUDA TRUE)
    find_package(CUDAToolkit REQUIRED)
  else()
    message(WARNING "CUDA not found; building CPU-only stubs. Set -DTRELLIS_WITH_CUDA=OFF to hide this warning.")
    set(HAVE_CUDA FALSE)
  endif()
else()
  set(HAVE_CUDA FALSE)
endif()

# Optional zlib for NPZ DEFLATE
find_package(ZLIB)
if (ZLIB_FOUND)
  message(STATUS "ZLIB found: ${ZLIB_VERSION}")
  set(HAVE_ZLIB TRUE)
else()
  message(STATUS "ZLIB not found: NPZ DEFLATE entries will not be supported")
  set(HAVE_ZLIB FALSE)
endif()

# Sources
set(RUNTIME_SOURCES
  runtime/tensor.hpp
  runtime/allocator.hpp
  runtime/allocator.cpp
  runtime/device.hpp
  runtime/device.cpp
  runtime/gemm.hpp
  runtime/gemm.cu
  runtime/cast.hpp
  runtime/cast.cu
  runtime/transpose.hpp
  runtime/transpose.cu
  runtime/norm.hpp
  runtime/norm.cu
  runtime/graphs.hpp
  runtime/graphs.cu
  runtime/ops.hpp
  runtime/ops.cu
  runtime/safetensors.hpp
  runtime/safetensors.cpp
  runtime/npz.hpp
  runtime/npz.cpp
  runtime/json.hpp
  runtime/json.cpp
  runtime/npy_write.hpp
  runtime/npy_write.cpp
  common.hpp
)

# Model sources
set(MODEL_SOURCES
  models/mlp.hpp
  models/mlp.cpp
  models/attention.hpp
  models/attention.cu
  models/dit_block.hpp
  models/dit_block.cu
  models/ss_decoder.hpp
  models/ss_decoder.cu
  models/conv3d.hpp
  models/conv3d.cu
  models/ss_decoder_conv.hpp
  models/ss_decoder_conv.cu
)

set(SAMPLER_SOURCES
  samplers/flow_euler.hpp
  samplers/flow_euler.cpp
)

set(PIPELINE_SOURCES
  pipeline/config.hpp
  pipeline/config.cpp
  pipeline/bindings.hpp
  pipeline/bindings.cpp
  pipeline/trellis_image_to_3d.hpp
  pipeline/trellis_image_to_3d.cpp
)

# C API (shared library)
set(CAPI_SOURCES
  capi/trellis_c_api.hpp
  capi/trellis_c_api.cpp
)

add_library(trellis_runtime STATIC
  ${RUNTIME_SOURCES}
  ${MODEL_SOURCES}
  ${SAMPLER_SOURCES}
  ${PIPELINE_SOURCES}
)
target_include_directories(trellis_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(trellis_runtime PUBLIC $<$<BOOL:${HAVE_CUDA}>:TRELLIS_HAVE_CUDA=1> $<$<BOOL:${HAVE_ZLIB}>:TRELLIS_HAVE_ZLIB=1>)
if (ZLIB_FOUND)
  target_link_libraries(trellis_runtime PUBLIC ZLIB::ZLIB)
endif()

# CUDA kernels (optional)
add_library(trellis_kernels INTERFACE)

target_link_libraries(trellis_runtime PUBLIC trellis_kernels)
if (HAVE_CUDA)
  target_link_libraries(trellis_runtime PUBLIC CUDA::cudart)
  target_link_libraries(trellis_runtime PUBLIC CUDA::cublas)
endif()

# Shared library exporting the C API symbol(s)
add_library(trellis_infer SHARED
  ${CAPI_SOURCES}
)
target_link_libraries(trellis_infer PRIVATE trellis_runtime)
target_link_libraries(trellis_infer PRIVATE trellis_kernels)
target_include_directories(trellis_infer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# App
add_executable(trellis_infer_cli
  app/main.cpp
)
target_link_libraries(trellis_infer_cli PRIVATE trellis_runtime)
target_link_libraries(trellis_infer_cli PRIVATE trellis_kernels)
if (HAVE_CUDA)
  set_target_properties(trellis_infer_cli PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(trellis_infer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

install(TARGETS trellis_infer_cli RUNTIME DESTINATION bin)
install(TARGETS trellis_infer LIBRARY DESTINATION lib)
