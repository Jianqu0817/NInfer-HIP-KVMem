// Uses real HIP buffers and the production pre-RoPE statistics adapter; no model is required.
#include "core/arena.h"
#include "core/device.h"
#include "models/qwen3_5/execution/memory_statistics.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
using ninfer::models::qwen3_5::execution::MemoryStatistics;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

std::uint16_t bf16(float value) {
    return std::uint16_t(std::bit_cast<std::uint32_t>(value) >> 16);
}

struct CaptureBuffers {
    DeviceBuffer queries, keys, positions;
    Tensor q, k, p;

    explicit CaptureBuffers(std::uint32_t rows)
        : queries(rows * 4 * sizeof(std::uint16_t)), keys(rows * 2 * sizeof(std::uint16_t)),
          positions(rows * sizeof(std::int32_t)),
          q(queries.p, DType::BF16, {2, 2, std::int32_t(rows)}),
          k(keys.p, DType::BF16, {2, 1, std::int32_t(rows)}),
          p(positions.p, DType::I32, {std::int32_t(rows)}) {}

    void upload(const std::vector<float>& query, const std::vector<float>& key,
                const std::vector<std::int32_t>& offsets) {
        require(query.size() == std::size_t(q.numel()) && key.size() == std::size_t(k.numel()) &&
                    offsets.size() == std::size_t(p.numel()), "bad statistics fixture shape");
        std::vector<std::uint16_t> query_bits(query.size()), key_bits(key.size());
        for (std::size_t i = 0; i < query.size(); ++i) query_bits[i] = bf16(query[i]);
        for (std::size_t i = 0; i < key.size(); ++i) key_bits[i] = bf16(key[i]);
        queries.copy_from_host(query_bits.data(), query_bits.size() * sizeof(std::uint16_t));
        keys.copy_from_host(key_bits.data(), key_bits.size() * sizeof(std::uint16_t));
        positions.copy_from_host(offsets.data(), offsets.size() * sizeof(std::int32_t));
    }
};

void expect(const std::vector<double>& actual, const std::vector<double>& expected,
            const char* message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-9) {
            std::fprintf(stderr, "%s: page=%zu actual=%.12g expected=%.12g\n",
                         message, i, actual[i], expected[i]);
            throw std::runtime_error(message);
        }
    }
}
}

int main() {
    try {
        check(cudaSetDevice(0));
        MemoryStatistics statistics(1, 2, 2, 1, 256, 0.5);
        CaptureBuffers prefix(128);
        std::vector<float> query(128 * 4), key(128 * 2);
        std::vector<std::int32_t> positions(128);
        for (std::uint32_t row = 0; row < 128; ++row) {
            query[row * 4] = 1;
            query[row * 4 + 2] = 1;
            key[row * 2 + (row < 64 ? 0 : 1)] = 1;
            positions[row] = std::int32_t(row);
        }
        prefix.upload(query, key, positions);
        statistics.begin_query();
        statistics.capture(0, prefix.q, prefix.k, prefix.p, 0, nullptr);
        statistics.commit(128);

        // Compact positions 64/65 map back to logical 128/129. The repeated 64 is an
        // invalid padded verification column, and logical 129 is an unaccepted candidate.
        CaptureBuffers candidates(3);
        candidates.upload({1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1},
                          {-1, 0, 0, 1, 100, 0}, {64, 64, 65});
        statistics.begin_query();
        statistics.capture(0, candidates.q, candidates.k, candidates.p, 64, nullptr);
        statistics.commit(129);
        const double cosine = 1.0 / (1.0 + 1e-6);
        const std::vector<double> first{cosine, 0, -cosine};
        expect(statistics.scores(), first, "signed cosine or accepted-prefix filtering");
        statistics.commit_score_preview();
        expect(statistics.scores(true), first, "first LRFU publication");
        expect(statistics.scores(true), first, "repeated LRFU publication changed one epoch");
        statistics.clear_query();
        expect(statistics.scores(true), first, "replay reset the LRFU epoch");
        require(statistics.captured_rows() == 130, "duplicate candidate entered statistics");

        CaptureBuffers next(1);
        next.upload({-1, 0, -1, 0}, {-1, 0}, {65});
        statistics.begin_query();
        statistics.capture(0, next.q, next.k, next.p, 64, nullptr);
        statistics.commit(130);
        expect(statistics.scores(true), {-0.5 * cosine, 0, 0.5 * cosine},
               "second query LRFU decay or mean-key aggregation");

        // Independent GQA oracle: mean([1,0], [0,1]) = [0.5,0.5]. Both positive
        // basis keys have the same cosine; taking a max across Q heads would give 1.
        MemoryStatistics gqa(1, 2, 2, 1, 256, 0.5);
        gqa.capture(0, prefix.q, prefix.k, prefix.p, 0, nullptr);
        gqa.commit(128);
        next.upload({1, 0, 0, 1}, {-1, 0}, {128});
        gqa.begin_query();
        gqa.capture(0, next.q, next.k, next.p, 0, nullptr);
        gqa.commit(129);
        const double mean_cosine = 0.5 / (std::sqrt(0.5) + 1e-6);
        expect(gqa.scores(), {mean_cosine, mean_cosine, -mean_cosine},
               "GQA query-head averaging");
        std::puts("KVMem pre-RoPE BF16 statistics and accepted-prefix LRFU: PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMem statistics GPU test failed: %s\n", error.what());
        return 1;
    }
}
