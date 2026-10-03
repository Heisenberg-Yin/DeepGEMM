#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"
#include <deep_gemm/mqa_bf16/layout.cuh>

namespace deep_gemm::mqa_bf16 {

// A paired API: the BF16 schedule uses split 384 and must never consume the
// legacy paged API's split-256 metadata. The hint is CPU configuration; request
// runs and causal lengths remain on the device.
struct Config {
    int block_q, num_tmem_stages, num_kv_stages;
    bool histogram, swizzle;
    static constexpr int split_kv = 384;

    Config(int tokens_per_request, bool histogram):
        block_q(tokens_per_request >= 5 ? 6 : 4),
        num_tmem_stages(tokens_per_request >= 5 ? 2 : 3),
        num_kv_stages(tokens_per_request >= 5 or histogram ? 7 : 8),
        histogram(histogram), swizzle(tokens_per_request == 6) {
        DG_HOST_ASSERT(tokens_per_request >= 1 and tokens_per_request <= 6);
    }

    int storage_size() const {
        using FP4 = cutlass::float_e2m1_t;
        if (block_q == 6)
            return sizeof(layout::MQALogitsSharedStorage<32, 128, 6, 384, 192, 2, 7, 2, FP4>);
        if (num_kv_stages == 7)
            return sizeof(layout::MQALogitsSharedStorage<32, 128, 4, 384, 128, 2, 7, 3, FP4>);
        return sizeof(layout::MQALogitsSharedStorage<32, 128, 4, 384, 128, 2, 8, 3, FP4>);
    }

    int smem_size() const {
        const int bytes = storage_size() + (histogram ? 8 * 1024 * sizeof(uint16_t) : 0);
        DG_HOST_ASSERT(bytes <= SM100ArchSpec::smem_capacity);
        return bytes;
    }
};

class MetadataRuntime final: public LaunchRuntime<MetadataRuntime> {
public:
    struct Args {
        int rows, num_sms, block_q, num_threads;
        int *lens, *indices, *schedule;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/mqa_bf16/scheduler.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::mqa_bf16::sched::sm100_paged_mqa_logits_metadata<384, {}, {}, {}>);
}}
)", a.num_sms, a.block_q, a.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, a.rows, a.lens, a.indices, a.schedule));
    }
};

static void metadata(const torch::Tensor& lens, const torch::Tensor& indices,
                     const torch::Tensor& schedule, int num_sms, int tokens_per_request) {
    const int rows = lens.size(0);
    const int threads = rows <= 512 ? 256 : (rows <= 2048 ? 512 : 1024);
    const int smem = (2 * rows + threads / 32 + 1) * sizeof(int);
    DG_HOST_ASSERT(smem <= SM100ArchSpec::smem_capacity);
    const MetadataRuntime::Args a = {
        rows, num_sms, Config(tokens_per_request, false).block_q, threads,
        lens.data_ptr<int>(), indices.data_ptr<int>(), schedule.data_ptr<int>(),
        LaunchArgs(1, threads, smem)
    };
    const auto kernel = compiler->build("sm100_paged_mqa_logits_bf16_metadata", MetadataRuntime::generate(a));
    MetadataRuntime::launch(kernel, a);
}

class PagedRuntime final: public LaunchRuntime<PagedRuntime> {
public:
    struct Args {
        Config config;
        int rows, logits_stride, table_stride;
        int *lens, *table, *indices, *schedule, *histogram;
        void* logits;
        CUtensorMap q, sf_q, kv, sf_kv, weights;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& a) {
        const auto& c = a.config;
        return fmt::format(R"(
#include <deep_gemm/mqa_bf16/kernel.cuh>
using namespace deep_gemm::mqa_bf16;
static_assert(sizeof(layout::MQALogitsSharedStorage<32, 128, {}, 384, {}, 2, {}, {}, cutlass::float_e2m1_t>) == {});
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_paged_mqa_logits<
        32, 128, 128, {}, {}, 2, {}, {}, 384, {}, 128, 384, cutlass::float_e2m1_t, {}, {}
    >);
}}
)", c.block_q, c.block_q * 32, c.num_kv_stages, c.num_tmem_stages, c.storage_size(),
    c.block_q, c.block_q * 32, c.num_kv_stages, c.num_tmem_stages, c.num_kv_stages,
    c.histogram, c.swizzle);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            a.rows, a.logits_stride, a.table_stride, a.lens, a.logits, a.table, a.indices, a.schedule,
            a.q, a.sf_q, a.kv, a.sf_kv, a.weights, a.histogram));
    }
};

static void paged(const Config& c, const torch::Tensor& q, const torch::Tensor& sf_q,
                  const torch::Tensor& kv, const torch::Tensor& sf_kv, const torch::Tensor& weights,
                  const torch::Tensor& lens, const torch::Tensor& table, const torch::Tensor& indices,
                  const torch::Tensor& schedule, const torch::Tensor& logits, int* histogram) {
    const int rows = q.size(0), num_sms = device_runtime->get_num_sms();
    const PagedRuntime::Args a = {
        c, rows, static_cast<int>(logits.stride(0)), static_cast<int>(table.stride(0)),
        lens.data_ptr<int>(), table.data_ptr<int>(), indices.data_ptr<int>(), schedule.data_ptr<int>(),
        histogram, logits.data_ptr(),
        make_tma_2d_desc(q, 128, rows * 32, 128, c.block_q * 32, static_cast<int>(q.stride(2)), 64, 0, false, false),
        make_tma_2d_desc(sf_q, 32, rows, 32, c.block_q, static_cast<int>(sf_q.stride(1)), 0),
        make_tma_3d_desc(kv, 128, 128, kv.size(0), 128, 128, 1,
                         static_cast<int>(kv.stride(1)), static_cast<int>(kv.stride(0)), 64, 0, false, false),
        make_tma_2d_desc(sf_kv, 128, sf_kv.size(0), 128, 1, static_cast<int>(sf_kv.stride(0)), 0),
        make_tma_2d_desc(weights, 32, rows, 32, c.block_q, static_cast<int>(weights.stride(0)), 0),
        LaunchArgs(num_sms, 512, c.smem_size())
    };
    const auto kernel = compiler->build("sm100_paged_mqa_logits_bf16", PagedRuntime::generate(a));
    PagedRuntime::launch(kernel, a);
}

} // namespace deep_gemm::mqa_bf16
