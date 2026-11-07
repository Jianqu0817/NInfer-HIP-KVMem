#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NINFER_MTP_ROUND_HD __host__ __device__
#else
#define NINFER_MTP_ROUND_HD
#endif

namespace ninfer { namespace ops { namespace detail {

struct MtpRoundContinuation {
    std::int32_t cache_frontier;
    std::int32_t next_extent;
};

// Acceptance advances the logical frontier. The compact frontier remains the
// ingress value and advances independently by the number of licensed inputs.
NINFER_MTP_ROUND_HD inline MtpRoundContinuation mtp_round_continuation(
    std::int32_t base_cache_frontier, std::int32_t updated_logical_frontier,
    std::int32_t licensed, std::int32_t remaining_budget, std::int32_t drafts,
    std::int32_t logical_capacity, std::int32_t cache_capacity) {
    const auto cache_frontier = base_cache_frontier + licensed;
    auto next = remaining_budget - licensed - 1;
    const auto logical_room = logical_capacity - updated_logical_frontier - 1;
    const auto cache_room = cache_capacity - cache_frontier - 1;
    next = next < logical_room ? next : logical_room;
    next = next < cache_room ? next : cache_room;
    next = next < 0 ? 0 : (next > drafts ? drafts : next);
    return {cache_frontier, next};
}

}}} // namespace ninfer::ops::detail

#undef NINFER_MTP_ROUND_HD
