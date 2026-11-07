#include "ops/kernel/mtp_round_state.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using ninfer::ops::detail::mtp_round_continuation;

static void expect(std::int32_t actual, std::int32_t wanted, const char* label) {
    if (actual != wanted) {
        std::fprintf(stderr, "%s: got %d, expected %d\n", label, actual, wanted);
        std::exit(1);
    }
}

int main() {
    // The compact history has 64 columns, but the accepted history is at 100000.
    // Zero, one, and both drafts accepted must advance past the old anchor too.
    for (std::int32_t accepted = 0; accepted <= 2; ++accepted) {
        const auto licensed = accepted + 1;
        const auto round = mtp_round_continuation(64, 100000 + licensed,
                                                  licensed, 20, 2, 131072, 128);
        expect(round.cache_frontier, 65 + accepted, "anchor plus accepted drafts");
        expect(round.next_extent, 2, "next draft window");
        expect(round.cache_frontier + (100000 - 64), 100001 + accepted,
               "AR absolute RoPE position");
    }

    auto round = mtp_round_continuation(64, 131071, 2, 20, 2, 131072, 128);
    expect(round.next_extent, 0, "logical capacity leaves anchor only");
    round = mtp_round_continuation(64, 131070, 2, 20, 2, 131072, 128);
    expect(round.next_extent, 1, "logical capacity leaves one draft");
    round = mtp_round_continuation(126, 100001, 1, 20, 2, 131072, 128);
    expect(round.next_extent, 0, "physical capacity leaves anchor only");
    round = mtp_round_continuation(125, 100001, 1, 20, 2, 131072, 128);
    expect(round.next_extent, 1, "physical capacity leaves one draft");
    round = mtp_round_continuation(64, 100003, 3, 4, 2, 131072, 128);
    expect(round.next_extent, 0, "output budget leaves anchor only");

    // A full physical tail produces no proposals. After a safe-boundary view
    // change, the stale drafts are cleared, so extent-zero verification licenses
    // just the current anchor. It must still rebuild a complete proposal window.
    round = mtp_round_continuation(125, 100003, 3, 20, 2, 131072, 128);
    expect(round.next_extent, 0, "full physical tail");
    round = mtp_round_continuation(64, 100004, 1, 17, 2, 131072, 128);
    expect(round.cache_frontier, 65, "reselected anchor-only verification");
    expect(round.next_extent, 2, "reselected view restores proposals");

    // Exhaustively compare with the old dense-path contract when cache/logical
    // positions coincide. This catches double-advancing the already updated E.
    for (std::int32_t capacity = 2; capacity <= 130; ++capacity) {
        for (std::int32_t base = 0; base < capacity; ++base) {
            for (std::int32_t accepted = 0; accepted <= 2; ++accepted) {
                const auto licensed = accepted + 1;
                if (base + licensed > capacity) { continue; }
                for (std::int32_t budget = licensed; budget <= licensed + 5; ++budget) {
                    auto expected = budget - licensed - 1;
                    if (expected > capacity - base - licensed - 1) {
                        expected = capacity - base - licensed - 1;
                    }
                    if (expected < 0) { expected = 0; }
                    if (expected > 2) { expected = 2; }
                    round = mtp_round_continuation(base, base + licensed,
                                                    licensed, budget, 2, capacity, capacity);
                    expect(round.cache_frontier, base + licensed, "dense AR frontier");
                    expect(round.next_extent, expected, "dense-path equivalence");
                }
            }
        }
    }
    std::puts("MTP continuation arithmetic: PASS");
}
