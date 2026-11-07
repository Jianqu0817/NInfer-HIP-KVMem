#pragma once
// Adapted without arithmetic changes from melis2023/ninfer-kvmem-actkv
// kvmem/include/kvmem/lrfu.hpp, commit a60025a12546cf8a45e084836ae7a77e322fcbf0.
// This header retains only the RAM LRFU state and history; snapshot I/O is not used.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
namespace kvmem {
struct LrfuState {
    std::shared_ptr<const std::vector<double>> scores = std::make_shared<const std::vector<double>>();
    uint64_t round = 0;
    bool committed = false;
};

class LrfuHistory {
public:
    LrfuHistory(double lambda = 0.5, double top_fraction = 0.9)
        : lambda_(lambda), fraction_(top_fraction) {
        if (!std::isfinite(lambda) || lambda < 0 || lambda > 1 ||
            !std::isfinite(top_fraction) || top_fraction <= 0 || top_fraction > 1)
            throw std::invalid_argument("invalid LRFU decay or hit fraction");
    }
    void begin_round() {
        if (++state_.round == 0) throw std::overflow_error("LRFU round overflow");
        state_.committed = false;
        candidate_.reset();
    }
    // Repeated probes always derive from committed history, never another probe.
    void preview(const std::vector<double>& scores) {
        if (state_.committed) return;
        if (!state_.round) throw std::logic_error("LRFU probe outside a round");
        for (double value : scores) if (!std::isfinite(value))
            throw std::invalid_argument("nonfinite LRFU retrieval score");
        if (scores.empty()) { candidate_.reset(); return; }
        auto next = std::make_shared<std::vector<double>>(scores.size());
        double threshold = -INFINITY;
        if (fraction_ < 1) {
            auto ranked = scores;
            const auto rank = std::min(scores.size() - 1,
                static_cast<size_t>((1 - fraction_) * scores.size()));
            std::nth_element(ranked.begin(), ranked.begin() + rank, ranked.end());
            threshold = ranked[rank];
        }
        // The hit fraction gates positive reinforcement only. Dropping an
        // unselected negative score to zero would reward it relative to a hit
        // with a less-negative score, reversing relevance with no prior history.
        for (size_t i = 0; i < scores.size(); ++i)
            (*next)[i] = lambda_ * (i < state_.scores->size() ? (*state_.scores)[i] : 0) +
                (scores[i] >= threshold ? scores[i] : std::min(scores[i], 0.));
        candidate_ = std::move(next);
    }
    bool commit() noexcept {
        if (state_.committed || !candidate_) return false;
        state_.scores = std::move(candidate_);
        state_.committed = true;
        return true;
    }
    double score(size_t block) const noexcept {
        const auto& scores = candidate_ ? *candidate_ : *state_.scores;
        return block < scores.size() ? scores[block] : 0;
    }
    LrfuState checkpoint() const { return state_; }
    void restore(const LrfuState& state) {
        if (!state.scores || (!state.round && (state.committed || !state.scores->empty())))
            throw std::invalid_argument("invalid LRFU checkpoint");
        for (double value : *state.scores) if (!std::isfinite(value))
            throw std::invalid_argument("nonfinite LRFU checkpoint");
        state_ = state;
        candidate_.reset();
    }
    void discard_candidate() noexcept { candidate_.reset(); }
    size_t allocated_bytes() const noexcept {
        return state_.scores->capacity() * sizeof(double) +
            (candidate_ ? candidate_->capacity() * sizeof(double) : 0);
    }
private:
    double lambda_, fraction_;
    LrfuState state_;
    std::shared_ptr<const std::vector<double>> candidate_;
};

}
