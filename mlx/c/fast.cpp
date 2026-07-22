/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/fast.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/ops.h"

namespace {
auto& compiled_swiglu() {
  static auto compiled = mlx::core::compile(
      [](const std::vector<mlx::core::array>& inputs) {
        const auto& gate = inputs[0];
        const auto& x = inputs[1];
        auto activated = mlx::core::multiply(gate, mlx::core::sigmoid(gate));
        return std::vector<mlx::core::array>{
            mlx::core::multiply(activated, x)};
      },
      true);
  return compiled;
}
} // namespace

namespace {

constexpr const char* kLmHeadArgmaxPartialsSource = R"metal(
  constexpr uint kRowsPerThreadgroup = 32;
  constexpr uint kRowsPerSimdgroup = 4;
  constexpr uint kElementsPerThread = 4;
  constexpr uint kHiddenBlock = 128;

  uint lane = thread_index_in_simdgroup;
  uint simdgroup = simdgroup_index_in_threadgroup;
  uint group = threadgroup_position_in_grid.x;
  uint row_base = group * kRowsPerThreadgroup +
      simdgroup * kRowsPerSimdgroup;
  uint hidden_size = embedding_shape[1];
  uint vocab_size = embedding_shape[0];

  float sums[kRowsPerSimdgroup] = {0.0f};
  for (uint block = 0; block < hidden_size; block += kHiddenBlock) {
    uint column = block + lane * kElementsPerThread;
    for (uint element = 0; element < kElementsPerThread; ++element) {
      uint current = column + element;
      float coefficient = current < hidden_size
          ? static_cast<float>(hidden[current])
          : 0.0f;
      for (uint row = 0; row < kRowsPerSimdgroup; ++row) {
        uint vocab_row = row_base + row;
        float weight = current < hidden_size && vocab_row < vocab_size
            ? static_cast<float>(
                  embedding[vocab_row * hidden_size + current])
            : 0.0f;
        sums[row] += weight * coefficient;
      }
    }
  }

  for (uint offset = 16; offset > 0; offset >>= 1) {
    for (uint row = 0; row < kRowsPerSimdgroup; ++row) {
      sums[row] += simd_shuffle_down(sums[row], offset);
    }
  }

  threadgroup T block_values[kRowsPerThreadgroup];
  threadgroup uint block_indices[kRowsPerThreadgroup];
  if (lane == 0) {
    for (uint row = 0; row < kRowsPerSimdgroup; ++row) {
      uint slot = simdgroup * kRowsPerSimdgroup + row;
      uint vocab_row = row_base + row;
      block_values[slot] = static_cast<T>(sums[row]);
      block_indices[slot] = vocab_row;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup == 0) {
    T best_value = block_values[lane];
    uint best_index = block_indices[lane];
    for (uint offset = 16; offset > 0; offset >>= 1) {
      T candidate_value = simd_shuffle_down(best_value, offset);
      uint candidate_index = simd_shuffle_down(best_index, offset);
      if (best_value < candidate_value ||
          (best_value == candidate_value && best_index > candidate_index)) {
        best_value = candidate_value;
        best_index = candidate_index;
      }
    }
    if (lane == 0) {
      partial_values[group] = best_value;
      partial_indices[group] = best_index;
    }
  }
)metal";

constexpr const char* kLmHeadArgmaxFinalSource = R"metal(
  uint tid = thread_position_in_threadgroup.x;
  uint partial_count = partial_values_shape[0];
  T best_value = partial_values[tid];
  uint best_index = partial_indices[tid];
  for (uint index = tid + threads_per_threadgroup.x;
       index < partial_count;
       index += threads_per_threadgroup.x) {
    T candidate_value = partial_values[index];
    uint candidate_index = partial_indices[index];
    if (best_value < candidate_value ||
        (best_value == candidate_value && best_index > candidate_index)) {
      best_value = candidate_value;
      best_index = candidate_index;
    }
  }

  for (uint offset = 16; offset > 0; offset >>= 1) {
    T candidate_value = simd_shuffle_down(best_value, offset);
    uint candidate_index = simd_shuffle_down(best_index, offset);
    if (best_value < candidate_value ||
        (best_value == candidate_value && best_index > candidate_index)) {
      best_value = candidate_value;
      best_index = candidate_index;
    }
  }

  threadgroup T simd_values[8];
  threadgroup uint simd_indices[8];
  if (thread_index_in_simdgroup == 0) {
    simd_values[simdgroup_index_in_threadgroup] = best_value;
    simd_indices[simdgroup_index_in_threadgroup] = best_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_index_in_threadgroup == 0) {
    uint lane = thread_index_in_simdgroup;
    T group_value = lane < 8 ? simd_values[lane] : simd_values[0];
    uint group_index = lane < 8 ? simd_indices[lane] : simd_indices[0];
    for (uint offset = 16; offset > 0; offset >>= 1) {
      T candidate_value = simd_shuffle_down(group_value, offset);
      uint candidate_index = simd_shuffle_down(group_index, offset);
      if (group_value < candidate_value ||
          (group_value == candidate_value && group_index > candidate_index)) {
        group_value = candidate_value;
        group_index = candidate_index;
      }
    }
    if (lane == 0) {
      token[0] = static_cast<int>(group_index);
    }
  }
)metal";

const mlx::core::fast::CustomKernelFunction& lm_head_argmax_partials_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "lm_head_argmax_partials",
      {"hidden", "embedding"},
      {"partial_values", "partial_indices"},
      kLmHeadArgmaxPartialsSource,
      "",
      true,
      false);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction& lm_head_argmax_final_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "lm_head_argmax_final",
      {"partial_values", "partial_indices"},
      {"token"},
      kLmHeadArgmaxFinalSource,
      "",
      true,
      false);
  return kernel;
}

} // namespace

struct mlx_fast_cuda_kernel_config_cpp_ {
  std::vector<mlx::core::Shape> output_shapes;
  std::vector<mlx::core::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
      template_args;
  std::optional<float> init_value;
  bool verbose;
};

inline mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new_() {
  return mlx_fast_cuda_kernel_config({new mlx_fast_cuda_kernel_config_cpp_()});
}

inline mlx_fast_cuda_kernel_config_cpp_& mlx_fast_cuda_kernel_config_get_(
    mlx_fast_cuda_kernel_config d) {
  if (!d.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_fast_cuda_kernel_config");
  }
  return *static_cast<mlx_fast_cuda_kernel_config_cpp_*>(d.ctx);
}

inline void mlx_fast_cuda_kernel_config_free_(mlx_fast_cuda_kernel_config d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_cuda_kernel_config_cpp_*>(d.ctx);
  }
}

extern "C" int mlx_fast_compiled_swiglu(
    mlx_array* res,
    const mlx_array gate,
    const mlx_array x) {
  try {
    auto outputs = compiled_swiglu()(
        {mlx_array_get_(gate), mlx_array_get_(x)});
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_lm_head_argmax(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array embedding,
    const mlx_stream s) {
  try {
    const auto& hidden_array = mlx_array_get_(hidden);
    const auto& embedding_array = mlx_array_get_(embedding);
    const auto stream = mlx_stream_get_(s);
    const bool supported_dtype =
        hidden_array.dtype() == mlx::core::float32 ||
        hidden_array.dtype() == mlx::core::float16 ||
        hidden_array.dtype() == mlx::core::bfloat16;
    if (embedding_array.ndim() != 2 ||
        hidden_array.size() != embedding_array.shape(1) ||
        hidden_array.dtype() != embedding_array.dtype() ||
        !supported_dtype || embedding_array.shape(0) < 8192 ||
        embedding_array.shape(0) % 32 != 0) {
      throw std::invalid_argument(
          "[lm_head_argmax] expected matching FP32, FP16, or BF16 hidden=[H] and row-major embedding=[V,H], with V >= 8192 and divisible by 32.");
    }

    bool use_graph_fallback = stream.device == mlx::core::Device::cpu;
#if !defined(__APPLE__)
    use_graph_fallback = true;
#endif
    if (use_graph_fallback) {
      auto logits = mlx::core::matmul(
          hidden_array, mlx::core::transpose(embedding_array, stream), stream);
      logits = mlx::core::reshape(
          logits, {1, embedding_array.shape(0)}, stream);
      auto output = mlx::core::astype(
          mlx::core::argmax(logits, -1, false, stream),
          mlx::core::int32,
          stream);
      mlx_array_set_(*res, std::move(output));
      return 0;
    }

    const int partial_count = embedding_array.shape(0) / 32;
    auto partials = lm_head_argmax_partials_kernel()(
        {hidden_array, embedding_array},
        {{partial_count}, {partial_count}},
        {hidden_array.dtype(), mlx::core::uint32},
        {partial_count * 256, 1, 1},
        {256, 1, 1},
        {{"T", hidden_array.dtype()}},
        std::nullopt,
        false,
        stream);
    auto output = lm_head_argmax_final_kernel()(
        {partials.at(0), partials.at(1)},
        {{1}},
        {mlx::core::int32},
        {256, 1, 1},
        {256, 1, 1},
        {{"T", hidden_array.dtype()}},
        std::nullopt,
        false,
        stream);
    mlx_array_set_(*res, std::move(output.at(0)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void) {
  try {
    return mlx_fast_cuda_kernel_config_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_cuda_kernel_config_free(
    mlx_fast_cuda_kernel_config cls) {
  mlx_fast_cuda_kernel_config_free_(cls);
}

extern "C" int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).output_shapes.push_back(
        mlx::core::Shape(shape, shape + size));
    mlx_fast_cuda_kernel_config_get_(cls).output_dtypes.push_back(
        mlx_dtype_to_cpp(dtype));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).grid =
        std::make_tuple(grid1, grid2, grid3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).thread_group =
        std::make_tuple(thread1, thread2, thread3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).init_value = value;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).verbose = verbose;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), mlx_dtype_to_cpp(dtype)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_cuda_kernel_cpp_ {
  mlx::core::fast::CustomKernelFunction mkf;
  mlx_fast_cuda_kernel_cpp_(mlx::core::fast::CustomKernelFunction mkf)
      : mkf(mkf) {};
};

inline mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new_(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    int shared_memory) {
  return mlx_fast_cuda_kernel({new mlx_fast_cuda_kernel_cpp_(
      mlx::core::fast::cuda_kernel(
          name,
          input_names,
          output_names,
          source,
          header,
          ensure_row_contiguous,
          shared_memory))});
}

extern "C" mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory) {
  try {
    return mlx_fast_cuda_kernel_new_(
        name,
        mlx_vector_string_get_(input_names),
        mlx_vector_string_get_(output_names),
        source,
        header,
        ensure_row_contiguous,
        shared_memory);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

inline mlx::core::fast::CustomKernelFunction& mlx_fast_cuda_kernel_get_(
    mlx_fast_cuda_kernel d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_fast_cuda_kernel");
  }
  return static_cast<mlx_fast_cuda_kernel_cpp_*>(d.ctx)->mkf;
}

inline void mlx_fast_cuda_kernel_free_(mlx_fast_cuda_kernel d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_cuda_kernel_cpp_*>(d.ctx);
  }
}

extern "C" void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls) {
  mlx_fast_cuda_kernel_free_(cls);
}

extern "C" int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream) {
  try {
    auto config_ctx = mlx_fast_cuda_kernel_config_get_(config);
    mlx_vector_array_set_(
        *outputs,
        mlx_fast_cuda_kernel_get_(cls)(
            mlx_vector_array_get_(inputs),
            config_ctx.output_shapes,
            config_ctx.output_dtypes,
            config_ctx.grid,
            config_ctx.thread_group,
            config_ctx.template_args,
            config_ctx.init_value,
            config_ctx.verbose,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::layer_norm(
            mlx_array_get_(x),
            (weight.ctx ? std::make_optional(mlx_array_get_(weight))
                        : std::nullopt),
            (bias.ctx ? std::make_optional(mlx_array_get_(bias))
                      : std::nullopt),
            eps,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_metal_kernel_config_cpp_ {
  std::vector<mlx::core::Shape> output_shapes;
  std::vector<mlx::core::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
      template_args;
  std::optional<float> init_value;
  bool verbose;
};

inline mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new_() {
  return mlx_fast_metal_kernel_config(
      {new mlx_fast_metal_kernel_config_cpp_()});
}

inline mlx_fast_metal_kernel_config_cpp_& mlx_fast_metal_kernel_config_get_(
    mlx_fast_metal_kernel_config d) {
  if (!d.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_fast_metal_kernel_config");
  }
  return *static_cast<mlx_fast_metal_kernel_config_cpp_*>(d.ctx);
}

inline void mlx_fast_metal_kernel_config_free_(mlx_fast_metal_kernel_config d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_metal_kernel_config_cpp_*>(d.ctx);
  }
}

extern "C" mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void) {
  try {
    return mlx_fast_metal_kernel_config_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_metal_kernel_config_free(
    mlx_fast_metal_kernel_config cls) {
  mlx_fast_metal_kernel_config_free_(cls);
}

extern "C" int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).output_shapes.push_back(
        mlx::core::Shape(shape, shape + size));
    mlx_fast_metal_kernel_config_get_(cls).output_dtypes.push_back(
        mlx_dtype_to_cpp(dtype));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).grid =
        std::make_tuple(grid1, grid2, grid3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).thread_group =
        std::make_tuple(thread1, thread2, thread3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).init_value = value;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).verbose = verbose;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), mlx_dtype_to_cpp(dtype)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_metal_kernel_cpp_ {
  mlx::core::fast::CustomKernelFunction mkf;
  mlx_fast_metal_kernel_cpp_(mlx::core::fast::CustomKernelFunction mkf)
      : mkf(mkf) {};
};

inline mlx_fast_metal_kernel mlx_fast_metal_kernel_new_(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  return mlx_fast_metal_kernel({new mlx_fast_metal_kernel_cpp_(
      mlx::core::fast::metal_kernel(
          name,
          input_names,
          output_names,
          source,
          header,
          ensure_row_contiguous,
          atomic_outputs))});
}

extern "C" mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  try {
    return mlx_fast_metal_kernel_new_(
        name,
        mlx_vector_string_get_(input_names),
        mlx_vector_string_get_(output_names),
        source,
        header,
        ensure_row_contiguous,
        atomic_outputs);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

inline mlx::core::fast::CustomKernelFunction& mlx_fast_metal_kernel_get_(
    mlx_fast_metal_kernel d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_fast_metal_kernel");
  }
  return static_cast<mlx_fast_metal_kernel_cpp_*>(d.ctx)->mkf;
}

inline void mlx_fast_metal_kernel_free_(mlx_fast_metal_kernel d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_metal_kernel_cpp_*>(d.ctx);
  }
}

extern "C" void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls) {
  mlx_fast_metal_kernel_free_(cls);
}

extern "C" int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream) {
  try {
    auto config_ctx = mlx_fast_metal_kernel_config_get_(config);
    mlx_vector_array_set_(
        *outputs,
        mlx_fast_metal_kernel_get_(cls)(
            mlx_vector_array_get_(inputs),
            config_ctx.output_shapes,
            config_ctx.output_dtypes,
            config_ctx.grid,
            config_ctx.thread_group,
            config_ctx.template_args,
            config_ctx.init_value,
            config_ctx.verbose,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rms_norm(
            mlx_array_get_(x),
            (weight.ctx ? std::make_optional(mlx_array_get_(weight))
                        : std::nullopt),
            eps,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rope(
            mlx_array_get_(x),
            dims,
            traditional,
            (base.has_value ? std::make_optional<float>(base.value)
                            : std::nullopt),
            scale,
            offset,
            (freqs.ctx ? std::make_optional(mlx_array_get_(freqs))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_rope_dynamic(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    const mlx_array offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rope(
            mlx_array_get_(x),
            dims,
            traditional,
            (base.has_value ? std::make_optional<float>(base.value)
                            : std::nullopt),
            scale,
            mlx_array_get_(offset),
            (freqs.ctx ? std::make_optional(mlx_array_get_(freqs))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

namespace {

constexpr const char* kPaddleOcrRope2dQkSource = R"metal(
  uint dim = thread_position_in_grid.x;
  uint token = thread_position_in_grid.y;
  uint qk_head = thread_position_in_grid.z;
  uint kind = qk_head >> 4;
  uint head = qk_head & 15;
  uint input_base = token * 3456 + kind * 1152 + head * 72;
  uint pair = dim < 36 ? dim + 36 : dim - 36;
  float rotated = dim < 36 ? -qkv[input_base + pair] : qkv[input_base + pair];
  uint output_index = (qk_head * qkv_shape[0] + token) * 72 + dim;
  qk[output_index] = qkv[input_base + dim] * cosine[token * 72 + dim] +
      rotated * sine[token * 72 + dim];
)metal";

constexpr const char* kPaddleOcrRope2dQkBfloat16Source = R"metal(
  uint dim = thread_position_in_grid.x;
  uint token = thread_position_in_grid.y;
  uint qk_head = thread_position_in_grid.z;
  uint kind = qk_head >> 4;
  uint head = qk_head & 15;
  uint input_base = token * 3456 + kind * 1152 + head * 72;
  uint pair = dim < 36 ? dim + 36 : dim - 36;
  float rotated = dim < 36 ? -qkv[input_base + pair] : qkv[input_base + pair];
  uint output_index = (qk_head * qkv_shape[0] + token) * 72 + dim;
  qk[output_index] = static_cast<bfloat>(
      qkv[input_base + dim] * cosine[token * 72 + dim] +
      rotated * sine[token * 72 + dim]);
)metal";

const mlx::core::fast::CustomKernelFunction& paddleocr_rope_2d_qk_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "paddleocr_rope_2d_qk",
      {"qkv", "cosine", "sine"},
      {"qk"},
      kPaddleOcrRope2dQkSource,
      "",
      true,
      false);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction&
paddleocr_rope_2d_qk_float16_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "paddleocr_rope_2d_qk_float16",
      {"qkv", "cosine", "sine"},
      {"qk"},
      kPaddleOcrRope2dQkSource,
      "",
      true,
      false);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction&
paddleocr_rope_2d_qk_bfloat16_kernel() {
  static const auto kernel = mlx::core::fast::metal_kernel(
      "paddleocr_rope_2d_qk_bfloat16",
      {"qkv", "cosine", "sine"},
      {"qk"},
      kPaddleOcrRope2dQkBfloat16Source,
      "",
      true,
      false);
  return kernel;
}

void validate_paddleocr_rope_2d_qk(
    const mlx::core::array& qkv,
    const mlx::core::array& cosine,
    const mlx::core::array& sine) {
  const auto token_count = qkv.shape(0);
  const bool valid_trig_shape =
      (cosine.ndim() == 2 && sine.ndim() == 2 &&
       cosine.shape(0) == token_count && sine.shape(0) == token_count &&
       cosine.shape(1) == 72 && sine.shape(1) == 72) ||
      (cosine.ndim() == 3 && sine.ndim() == 3 &&
       cosine.shape(0) == token_count && sine.shape(0) == token_count &&
       cosine.shape(1) == 1 && sine.shape(1) == 1 &&
       cosine.shape(2) == 72 && sine.shape(2) == 72);
  const bool supported_dtype =
      qkv.dtype() == mlx::core::float32 ||
      qkv.dtype() == mlx::core::float16 ||
      qkv.dtype() == mlx::core::bfloat16;
  if (qkv.ndim() != 2 || qkv.shape(1) != 3456 || !valid_trig_shape ||
      !supported_dtype || cosine.dtype() != qkv.dtype() ||
      sine.dtype() != qkv.dtype()) {
    throw std::invalid_argument(
        "[paddleocr_rope_2d_qk] expected FP32, FP16, or BF16 qkv=[L,3456] and matching cosine/sine=[L,72] or [L,1,72].");
  }
}

} // namespace

extern "C" int mlx_fast_paddleocr_rope_2d_qk(
    mlx_array* res,
    const mlx_array qkv,
    const mlx_array cosine,
    const mlx_array sine,
    const mlx_stream s) {
  try {
    const auto& qkv_array = mlx_array_get_(qkv);
    const auto& cosine_array = mlx_array_get_(cosine);
    const auto& sine_array = mlx_array_get_(sine);
    validate_paddleocr_rope_2d_qk(qkv_array, cosine_array, sine_array);

    const auto token_count = qkv_array.shape(0);
    const auto& kernel = qkv_array.dtype() == mlx::core::float16
        ? paddleocr_rope_2d_qk_float16_kernel()
        : qkv_array.dtype() == mlx::core::bfloat16
        ? paddleocr_rope_2d_qk_bfloat16_kernel()
        : paddleocr_rope_2d_qk_kernel();
    auto outputs = kernel(
        {qkv_array, cosine_array, sine_array},
        {{2, 16, token_count, 72}},
        {qkv_array.dtype()},
        {72, token_count, 32},
        {32, 1, 1},
        {},
        std::nullopt,
        false,
        mlx_stream_get_(s));
    mlx_array_set_(*res, std::move(outputs.at(0)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::scaled_dot_product_attention(
            mlx_array_get_(queries),
            mlx_array_get_(keys),
            mlx_array_get_(values),
            scale,
            std::string(mask_mode),
            (mask_arr.ctx ? std::make_optional(mlx_array_get_(mask_arr))
                          : std::nullopt),
            (sinks.ctx ? std::make_optional(mlx_array_get_(sinks))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
