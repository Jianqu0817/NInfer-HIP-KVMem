#pragma once

#include "core/tensor.h"
#include "models/qwen3_5/execution/lrfu_history.h"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <optional>

namespace ninfer::models::qwen3_5::execution {

// Eager HIP path: capture the represented BF16 normalization output before RoPE.
// Candidate rows are retained until Program publishes the exact accepted prefix.
class MemoryStatistics {
public:
    MemoryStatistics(std::uint32_t layers, std::uint32_t head_dim, std::uint32_t query_heads,
                     std::uint32_t kv_heads, std::uint32_t capacity, double lambda);
    void capture(std::uint32_t layer, const Tensor& q, const Tensor& k,
                 const Tensor& cache_positions, std::uint32_t removed, cudaStream_t stream);
    void commit(std::uint32_t frontier);
    void discard_from(std::uint32_t frontier);
    void begin_query();
    void clear_query();
    void query_from_pending();
    std::uint32_t removed_tokens = 0;
    std::vector<double> scores(bool commit_lrfu = false);
    void commit_score_preview() noexcept;
    std::uint64_t captured_rows() const noexcept { return captured_rows_; }

private:
    struct Row {
        std::uint32_t layer, position;
        std::vector<float> q, k;
    };

    std::uint32_t layers_, dim_, qheads_, kheads_, capacity_, frontier_ = 0;
    double lambda_;
    kvmem::LrfuHistory history_;
    std::optional<kvmem::LrfuHistory> prepared_history_;
    bool query_committed_ = false;
    std::vector<float> key_sums_, query_sums_;
    std::vector<std::uint32_t> key_counts_, query_counts_;
    std::vector<Row> pending_;
    std::unordered_set<std::uint64_t> pending_positions_;
    std::uint64_t captured_rows_ = 0;
};

}
