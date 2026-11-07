#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Required pages are pinned first; relevance fills the remaining slots. Original
// logical ordering is restored before execution, including the final partial page.
inline std::vector<std::uint32_t> select_kvmem_pages(
    std::uint32_t frontier, std::uint32_t selected_pages, std::uint32_t sink_pages,
    std::uint32_t recent_pages, const std::vector<double>& scores) {
    if (!selected_pages || std::uint64_t(sink_pages) + std::max(1u, recent_pages) > selected_pages)
        throw std::invalid_argument("KVMem pinned pages exceed selected capacity");
    for (const auto score : scores)
        if (!std::isfinite(score)) throw std::invalid_argument("nonfinite KVMem retrieval score");
    const auto count = (frontier + 63) / 64;
    const auto keep = std::min(count, selected_pages);
    std::vector<std::uint32_t> selected;
    selected.reserve(keep);
    const auto add = [&](std::uint32_t id) {
        if (selected.size() < keep && std::find(selected.begin(), selected.end(), id) == selected.end())
            selected.push_back(id);
    };
    for (std::uint32_t id = 0; id < std::min(count, sink_pages); ++id) add(id);
    const auto recent = std::min(count, std::max(1u, recent_pages));
    for (auto id = count - recent; id < count; ++id) add(id);
    std::vector<std::uint32_t> ranked(count);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::stable_sort(ranked.begin(), ranked.end(), [&](auto a, auto b) {
        const double x = a < scores.size() ? scores[a] : 0;
        const double y = b < scores.size() ? scores[b] : 0;
        return x == y ? a > b : x > y;
    });
    for (const auto id : ranked) add(id);
    std::sort(selected.begin(), selected.end());
    return selected;
}

}
