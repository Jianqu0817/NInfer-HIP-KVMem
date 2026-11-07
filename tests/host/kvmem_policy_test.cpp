#include "models/qwen3_5/program/storage/kvmem_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

using ninfer::models::qwen3_5::detail::select_kvmem_pages;

static void require(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "KVMem policy: %s\n", message);
        std::exit(1);
    }
}

template <class Call>
static void rejects(Call call, const char* message) {
    bool rejected = false;
    try { call(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

int main() {
    require(select_kvmem_pages(512, 4, 1, 1,
        {-100., -.9, -.2, -.7, -.1, -.5, -.8, -100.}) ==
        std::vector<std::uint32_t>({0, 2, 4, 7}),
        "signed relevance must prefer the less-negative pages");
    require(select_kvmem_pages(512, 4, 1, 1, std::vector<double>(8, 0.)) ==
        std::vector<std::uint32_t>({0, 5, 6, 7}),
        "equal relevance chooses newer pages, then restores logical order");
    require(select_kvmem_pages(65, 4, 1, 2, {}) ==
        std::vector<std::uint32_t>({0, 1}), "short histories retain every page");
    require(select_kvmem_pages(0, 4, 1, 1, {}).empty(), "empty history has no page");

    constexpr std::uint32_t selected_budget = 384; // 24K selected + 4K append reserve.
    constexpr std::uint32_t reserve_tokens = 4096;
    constexpr std::uint32_t sinks = 2;
    constexpr std::uint32_t recent = 16;
    for (const auto frontier : {32768u, 64001u, 100000u, 131071u, 131072u}) {
        const auto pages = (frontier + 63u) / 64u;
        std::vector<double> scores(pages, -.8);
        // Each reselection recalls different old pages, while deliberately giving
        // the sink and recent tail poor scores so the test depends on pinning.
        const auto recalled = frontier % 97u + sinks;
        scores[recalled] = .95;
        scores[recalled + 1u] = .9;
        for (std::uint32_t id = pages - recent; id < pages; ++id) scores[id] = -100.;
        scores[0] = scores[1] = -100.;
        const auto selected = select_kvmem_pages(frontier, selected_budget, sinks, recent, scores);
        require(selected.size() == selected_budget, "selection stays within its physical budget");
        require(std::is_sorted(selected.begin(), selected.end()), "execution order stays logical");
        require(std::adjacent_find(selected.begin(), selected.end()) == selected.end(),
                "a logical page cannot occupy two physical slots");
        require(selected.front() == 0 && selected[1] == 1, "sink pages remain pinned");
        for (std::uint32_t id = pages - recent; id < pages; ++id)
            require(std::binary_search(selected.begin(), selected.end(), id), "recent pages remain pinned");
        require(selected.back() == pages - 1, "the partial or full tail is the final physical page");
        require(std::binary_search(selected.begin(), selected.end(), recalled), "old relevant page is recalled");
        require(selected == select_kvmem_pages(frontier, selected_budget, sinks, recent, scores),
                "reselection is deterministic");

        // Count represented committed columns from the selected logical IDs.
        // This geometric oracle checks the compact offset used for future writes.
        std::uint32_t represented = 0;
        for (auto id : selected) represented += std::min(64u, frontier - id * 64u);
        const auto removed = (pages - selected.size()) * 64u;
        require(frontier - removed == represented, "compact frontier equals represented columns");
        require((pages - 1u) * 64u - removed == (selected.size() - 1u) * 64u,
                "logical tail maps to the final selected page");
        const auto append_limit = std::min(reserve_tokens, 131072u - frontier);
        for (std::uint32_t appended = 0; appended <= append_limit; ++appended) {
            require(frontier + appended - removed == represented + appended,
                    "append positions continue after compact committed history");
            require(represented + appended <= selected_budget * 64u + reserve_tokens,
                    "append reserve remains bounded at 128K logical history");
        }
    }

    rejects([] { select_kvmem_pages(512, 0, 0, 0, {}); }, "zero selected budget is rejected");
    rejects([] { select_kvmem_pages(512, 2, 2, 0, {}); }, "tail needs space beside the sink");
    rejects([] { select_kvmem_pages(512, 3, 2, 2, {}); }, "pinned pages cannot exceed selected budget");
    rejects([] { select_kvmem_pages(512, 4, 1, 1,
        {std::numeric_limits<double>::quiet_NaN()}); }, "NaN relevance is rejected");
    rejects([] { select_kvmem_pages(512, 4, 1, 1,
        {std::numeric_limits<double>::infinity()}); }, "infinite relevance is rejected");
    std::puts("KVMem page selection and compact append geometry: PASS");
}
