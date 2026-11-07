#include "models/qwen3_5/execution/lrfu_history.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

static void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "LRFU: %s\n", message);
        std::exit(1);
    }
}

static void near(double actual, double expected, const char* message) {
    require(std::abs(actual - expected) < 1e-12, message);
}

int main() {
    kvmem::LrfuHistory history(0.5, 0.5);
    bool rejected = false;
    try { history.preview({1.0}); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected, "a probe needs an access boundary");

    history.begin_round();
    history.preview({-0.8, -0.2, 0.1, 0.2, 0.4, 0.9});
    near(history.score(0), -0.8, "negative misses must retain signed relevance");
    near(history.score(2), 0.0, "positive misses receive no reinforcement");
    near(history.score(3), 0.2, "threshold ties qualify");
    require(history.commit(), "first accepted access commits");
    const auto first = history.checkpoint();
    require(!history.commit(), "same access cannot commit twice");
    history.preview({100.0, 100.0});
    near(history.score(0), -0.8, "a committed access cannot be overwritten by replay");

    history.begin_round();
    history.preview({0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
    near(history.score(0), -0.2, "new hit combines with decayed committed state");
    history.preview({0.4, 0.4, 0.4, 0.4, 0.4, 0.4});
    near(history.score(0), 0.0, "repeated probes derive from the parent, not previous probe");
    history.discard_candidate();
    near(history.score(0), -0.8, "aborted probe preserves committed parent");
    require(!history.commit(), "discarded probe cannot commit");

    history.restore(first);
    history.begin_round();
    history.preview({0.4, 0.4, 0.4, 0.4, 0.4, 0.4});
    require(history.commit(), "restored branch can accept a new access");
    near(history.score(5), 0.85, "restored branch keeps its own history");
    near((*first.scores)[5], 0.9, "checkpoint remains immutable");

    history.begin_round();
    rejected = false;
    try { history.preview({std::numeric_limits<double>::quiet_NaN()}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "nonfinite relevance is rejected");
    std::puts("LRFU access/preview/restore: PASS");
}
