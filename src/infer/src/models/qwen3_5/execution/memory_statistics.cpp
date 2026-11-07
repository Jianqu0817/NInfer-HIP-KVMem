#include "models/qwen3_5/execution/memory_statistics.h"
#include "core/device.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

float bf16(std::uint16_t value) { return std::bit_cast<float>(std::uint32_t(value) << 16); }

}

MemoryStatistics::MemoryStatistics(std::uint32_t layers, std::uint32_t dim,
                                   std::uint32_t qheads, std::uint32_t kheads,
                                   std::uint32_t capacity, double lambda)
    : layers_(layers), dim_(dim), qheads_(qheads), kheads_(kheads), capacity_(capacity),
      lambda_(lambda), history_(lambda),
      key_sums_(std::size_t(layers) * ((capacity + 63) / 64) * dim * kheads),
      query_sums_(std::size_t(layers) * dim * qheads),
      key_counts_(std::size_t(layers) * ((capacity + 63) / 64)), query_counts_(layers) {
    if (!layers || !dim || !qheads || !kheads || qheads % kheads || !capacity ||
        !std::isfinite(lambda) || lambda < 0 || lambda > 1)
        throw std::invalid_argument("invalid KVMem statistics geometry");
}

void MemoryStatistics::capture(std::uint32_t layer, const Tensor& q, const Tensor& k,
                               const Tensor& positions, std::uint32_t removed,
                               cudaStream_t stream) {
    if (layer >= layers_ || q.dtype != DType::BF16 || k.dtype != DType::BF16 ||
        q.ne[0] != dim_ || q.ne[1] != qheads_ || k.ne[0] != dim_ || k.ne[1] != kheads_ ||
        q.ne[2] != k.ne[2] || !q.is_contiguous() || !k.is_contiguous())
        throw std::invalid_argument("invalid KVMem pre-RoPE capture");
    const auto rows = std::uint32_t(q.ne[2]);
    if (positions.dtype != DType::I32 || !positions.is_contiguous() || positions.numel() < rows)
        throw std::invalid_argument("invalid KVMem capture positions");
    std::vector<std::uint16_t> queries(std::size_t(rows) * dim_ * qheads_);
    std::vector<std::uint16_t> keys(std::size_t(rows) * dim_ * kheads_);
    std::vector<std::int32_t> offsets(rows);
    CUDA_CHECK(cudaMemcpyAsync(queries.data(), q.data, queries.size() * 2,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(keys.data(), k.data, keys.size() * 2,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(offsets.data(), positions.data, rows * 4,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::uint32_t t = 0; t < rows; ++t) {
        if (offsets[t] < 0 || std::uint64_t(offsets[t]) + removed >= capacity_)
            continue;
        const auto position = std::uint32_t(offsets[t]) + removed;
        const auto key = (std::uint64_t(layer) << 32) | position;
        if (!pending_positions_.insert(key).second)
            continue;
        Row row{layer, position, {}, {}};
        row.q.resize(dim_ * qheads_);
        row.k.resize(dim_ * kheads_);
        for (std::size_t i = 0; i < row.q.size(); ++i)
            row.q[i] = bf16(queries[t * row.q.size() + i]);
        for (std::size_t i = 0; i < row.k.size(); ++i)
            row.k[i] = bf16(keys[t * row.k.size() + i]);
        pending_.push_back(std::move(row));
        ++captured_rows_;
    }
}

void MemoryStatistics::begin_query() {
    query_committed_ = false;
    clear_query();
}

void MemoryStatistics::clear_query() {
    prepared_history_.reset();
    std::fill(query_sums_.begin(), query_sums_.end(), 0);
    std::fill(query_counts_.begin(), query_counts_.end(), 0);
}

void MemoryStatistics::commit(std::uint32_t frontier) {
    if (frontier == frontier_ && pending_.empty())
        return;
    clear_query();
    const auto blocks = (capacity_ + 63) / 64;
    for (const auto& row : pending_)
        if (row.position >= frontier_ && row.position < frontier) {
            const auto block = std::size_t(row.layer) * blocks + row.position / 64;
            auto* destination = key_sums_.data() + block * dim_ * kheads_;
            for (std::size_t i = 0; i < row.k.size(); ++i)
                destination[i] += row.k[i];
            ++key_counts_[block];
            auto* query = query_sums_.data() + std::size_t(row.layer) * dim_ * qheads_;
            for (std::size_t i = 0; i < row.q.size(); ++i)
                query[i] += row.q[i];
            ++query_counts_[row.layer];
        }
    frontier_ = std::max(frontier_, frontier);
    pending_.clear();
    pending_positions_.clear();
}

void MemoryStatistics::discard_from(std::uint32_t frontier) {
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                  [&](const Row& r) { return r.position >= frontier; }),
                    pending_.end());
    pending_positions_.clear();
    for (const auto& row : pending_)
        pending_positions_.insert((std::uint64_t(row.layer) << 32) | row.position);
}

void MemoryStatistics::query_from_pending() {
    clear_query();
    for (const auto& row : pending_) {
        auto* query = query_sums_.data() + std::size_t(row.layer) * dim_ * qheads_;
        for (std::size_t i = 0; i < row.q.size(); ++i)
            query[i] += row.q[i];
        ++query_counts_[row.layer];
    }
}

std::vector<double> MemoryStatistics::scores(bool commit_lrfu) {
    const auto blocks = (frontier_ + 63) / 64, stride = (capacity_ + 63) / 64;
    std::vector<double> result(blocks);
    for (std::uint32_t b = 0; b < blocks; ++b) {
        double score = 0;
        std::uint32_t count = 0;
        for (std::uint32_t l = 0; l < layers_; ++l) {
            if (!key_counts_[l * stride + b] || !query_counts_[l])
                continue;
            for (std::uint32_t h = 0; h < kheads_; ++h) {
                double dot = 0, qn = 0, kn = 0;
                const auto* key = key_sums_.data() +
                                  (std::size_t(l) * stride + b) * dim_ * kheads_ + h * dim_;
                const auto* query = query_sums_.data() + std::size_t(l) * dim_ * qheads_;
                for (std::uint32_t d = 0; d < dim_; ++d) {
                    double qv = 0;
                    for (std::uint32_t g = 0; g < qheads_ / kheads_; ++g)
                        qv += query[(h * (qheads_ / kheads_) + g) * dim_ + d];
                    qv /= double(query_counts_[l]) * (qheads_ / kheads_);
                    const double kv = key[d] / double(key_counts_[l * stride + b]);
                    dot += qv * kv;
                    qn += qv * qv;
                    kn += kv * kv;
                }
                score += dot / (std::sqrt(qn * kn) + 1e-6);
                ++count;
            }
        }
        result[b] = count ? score / count : 0;
    }
    kvmem::LrfuHistory preview(lambda_);
    preview.restore(history_.checkpoint());
    preview.begin_round();
    preview.preview(result);
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = preview.score(i);
    prepared_history_ = std::move(preview);
    if (commit_lrfu) {
        commit_score_preview();
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = history_.score(i);
    }
    return result;
}

void MemoryStatistics::commit_score_preview() noexcept {
    if (query_committed_ || !prepared_history_)
        return;
    history_ = std::move(*prepared_history_);
    history_.commit();
    prepared_history_.reset();
    query_committed_ = true;
}

}
