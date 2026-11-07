#include "models/qwen3_5/program/storage/kvmem_window.h"
#include "models/qwen3_5/program/storage/kvmem_policy.h"
#include "models/qwen3_5/program/program_impl.h"
#include "core/device.h"
#include <algorithm>
#include <array>
#include <numeric>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

std::uint32_t blocks(std::uint32_t n) { return (n + 63) / 64; }

}

KvmemWindow::KvmemWindow(KvmemOptions options, std::uint32_t capacity,
                         const KVPageGeometry& main, const KVPageGeometry* mtp)
    : options_(options), main_layout_(plan_host_kv_page_layout(main)),
      main_archive_(blocks(capacity)), mtp_archive_(mtp ? blocks(capacity) : 0),
      valid_(blocks(capacity)) {
    if (main.page_tokens != 64 || (mtp && mtp->page_tokens != 64))
        throw std::invalid_argument("KVMem requires native 64-token pages");
    if (!capacity || !options.selected_tokens || !options.reserve_tokens ||
        options.selected_tokens % 64 || options.reserve_tokens % 64 ||
        options.sink_tokens % 64 || options.recent_tokens % 64 ||
        std::uint64_t(options.selected_tokens) + options.reserve_tokens > capacity ||
        std::uint64_t(options.sink_tokens) + std::max(64u, options.recent_tokens) >
            options.selected_tokens)
        throw std::invalid_argument("invalid KVMem window geometry or budget");
    pages_.reserve(blocks(capacity));
    if (mtp)
        mtp_layout_ = plan_host_kv_page_layout(*mtp);
    const std::uint64_t bytes = std::uint64_t(blocks(capacity)) *
                                (main_layout_.page_stride + mtp_layout_.page_stride);
    if (bytes > options.host_capacity_bytes)
        throw std::invalid_argument("KVMem native history exceeds configured host capacity");
    std::array<HostKVPageLayout, 2> layouts{main_layout_, mtp_layout_};
    arena_ = std::make_unique<HostKVArena>(main_layout_.page_stride + mtp_layout_.page_stride,
                                          std::span(layouts).first(mtp ? 2 : 1));
    auto main_staging = arena_->allocate(main_layout_, 1);
    if (!main_staging)
        throw std::runtime_error("KVMem main staging allocation failed");
    main_staging_ = std::move(*main_staging);
    if (mtp) {
        auto mtp_staging = arena_->allocate(mtp_layout_, 1);
        if (!mtp_staging)
            throw std::runtime_error("KVMem MTP staging allocation failed");
        mtp_staging_ = std::move(*mtp_staging);
    }
}

std::uint32_t KvmemWindow::compact_position(std::uint32_t logical) const {
    if (invalid_ || logical < removed_)
        throw std::logic_error("KVMem invalid logical position");
    return logical - removed_;
}

void KvmemWindow::commit(std::uint32_t frontier) {
    if (invalid_ || frontier < frontier_ || blocks(frontier) > valid_.size())
        throw std::logic_error("KVMem commit is outside its valid history");
    for (auto id = blocks(frontier_); id < blocks(frontier); ++id)
        pages_.push_back(id);
    frontier_ = frontier;
}

bool KvmemWindow::prepare(std::uint32_t frontier, std::uint32_t end, KvmemBinding main,
                          const KvmemBinding* mtp, const std::vector<double>& scores,
                          cudaStream_t stream, bool force) {
    if (invalid_ || frontier != frontier_ || end < frontier || blocks(end) > valid_.size() ||
        bool(mtp) != !mtp_archive_.empty())
        throw std::logic_error("KVMem prepare is outside committed boundary");
    if (!force && compact_position(end) <= options_.selected_tokens + options_.reserve_tokens)
        return false;
    if (end - frontier > options_.reserve_tokens)
        throw std::invalid_argument("KVMem append exceeds reserve");
    // Allocate pageable history before the destructive boundary; only one native page per pool is pinned.
    for (auto id : pages_) {
        if (main_archive_[id].empty())
            main_archive_[id].resize(main_layout_.page_stride);
        if (mtp && mtp_archive_[id].empty())
            mtp_archive_[id].resize(mtp_layout_.page_stride);
    }
    for (std::uint32_t slot = 0; slot < pages_.size(); ++slot) {
        const auto id = pages_[slot], columns = std::min(64u, frontier - id * 64);
        if (valid_[id] == columns)
            continue;
        const std::array source{main.addresses->physical_page(main.address, slot)};
        main.pages->physical_pool().copy_to_host(source, arena_->writable_view(main_staging_),
                                                 stream);
        if (mtp) {
            const std::array source_mtp{mtp->addresses->physical_page(mtp->address, slot)};
            mtp->pages->physical_pool().copy_to_host(source_mtp,
                                                     arena_->writable_view(mtp_staging_), stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::memcpy(main_archive_[id].data(), arena_->view(main_staging_).data(),
                    main_layout_.page_stride);
        if (mtp)
            std::memcpy(mtp_archive_[id].data(), arena_->view(mtp_staging_).data(),
                        mtp_layout_.page_stride);
        valid_[id] = columns;
        spilled_ += main_layout_.page_stride + mtp_layout_.page_stride;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto count = blocks(frontier);
    auto selected = select_kvmem_pages(frontier, options_.selected_tokens / 64,
                                       options_.sink_tokens / 64, options_.recent_tokens / 64,
                                       scores);
    if (selected == pages_)
        return false;
    for (auto id : selected)
        if (!valid_[id] || main_archive_[id].empty() || (mtp && mtp_archive_[id].empty()))
            throw std::logic_error("KVMem selected page has no committed native archive");
    try {
        main.addresses->destructive_truncate(main.address, 0);
        if (mtp)
            mtp->addresses->destructive_truncate(mtp->address, 0);
        const auto visible = frontier - (count - selected.size()) * 64;
        main.addresses->ensure_mapped_to_tokens(main.address, visible + end - frontier, stream);
        if (mtp)
            mtp->addresses->ensure_mapped_to_tokens(mtp->address, visible + end - frontier,
                                                    stream);
        for (std::uint32_t slot = 0; slot < selected.size(); ++slot) {
            const auto id = selected[slot];
            const std::array destination{main.addresses->physical_page(main.address, slot)};
            std::memcpy(arena_->writable_view(main_staging_).data(), main_archive_[id].data(),
                        main_layout_.page_stride);
            main.pages->physical_pool().copy_from_host(arena_->view(main_staging_), destination,
                                                       stream);
            if (mtp) {
                const std::array destination_mtp{mtp->addresses->physical_page(mtp->address, slot)};
                std::memcpy(arena_->writable_view(mtp_staging_).data(), mtp_archive_[id].data(),
                            mtp_layout_.page_stride);
                mtp->pages->physical_pool().copy_from_host(arena_->view(mtp_staging_),
                                                           destination_mtp, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            restored_ += main_layout_.page_stride + mtp_layout_.page_stride;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        main.addresses->commit_frontier(main.address, visible);
        if (mtp)
            mtp->addresses->commit_frontier(mtp->address, visible);
        pages_ = std::move(selected);
        removed_ = frontier - visible;
        ++swaps_;
        static const bool trace = [] {
            const char* value = std::getenv("NINFER_KVMEM_TRACE");
            return value != nullptr && value[0] == '1' && value[1] == '\0';
        }();
        if (trace) {
            std::fprintf(stderr,
                         "[kvmem] logical=%u resident=%u swaps=%llu spilled_bytes=%llu "
                         "restored_bytes=%llu\n",
                         frontier, static_cast<unsigned>(visible),
                         static_cast<unsigned long long>(swaps_),
                         static_cast<unsigned long long>(spilled_),
                         static_cast<unsigned long long>(restored_));
            std::fflush(stderr);
        }
    } catch (...) {
        invalid_ = true;
        throw;
    }
    return true;
}

std::uint32_t ProgramImpl::compact_position(const SequenceState& sequence,
                                           std::uint32_t logical) const {
    return sequence.window ? sequence.window->compact_position(logical) : logical;
}

bool ProgramImpl::prepare_memory_window(SequenceState& sequence, std::uint32_t begin,
                                       std::uint32_t end, bool force) {
    if (!kvmem.enabled)
        return false;
    if (!sequence.kv || begin != sequence.text_kv_valid ||
        (sequence.kv->backend && begin != sequence.mtp_kv_valid))
        throw std::logic_error("KVMem requires a committed main/MTP safe boundary");
    if (!sequence.window) {
        const auto* mtp_geometry =
            backend_kv_pages ? &backend_kv_pages->physical_pool().geometry() : nullptr;
        auto window = std::make_unique<KvmemWindow>(kvmem, capacity,
                                                   text_kv_pages->physical_pool().geometry(),
                                                   mtp_geometry);
        window->commit(begin);
        const auto& config = parameters.model.config().text;
        auto statistics = std::make_unique<execution::MemoryStatistics>(
            config.full_attention_layers, config.attention->head_dim,
            config.attention->num_attention_heads, config.attention->num_key_value_heads,
            capacity, kvmem.lrfu_lambda);
        sequence.window = std::move(window);
        sequence.statistics = std::move(statistics);
    }
    if (!force && compact_position(sequence, end) <= kvmem.selected_tokens + kvmem.reserve_tokens)
        return false;
    KvmemBinding main{text_kv_pages.get(), text_kv_addresses.get(), sequence.kv->text};
    std::optional<KvmemBinding> mtp;
    if (sequence.kv->backend)
        mtp = KvmemBinding{backend_kv_pages.get(), backend_kv_addresses.get(), *sequence.kv->backend};
    if (force)
        sequence.statistics->query_from_pending();
    const bool changed = sequence.window->prepare(begin, end, main, mtp ? &*mtp : nullptr,
                                                   sequence.statistics->scores(false),
                                                   device.stream, force);
    if (force)
        sequence.statistics->commit_score_preview();
    sequence.statistics->removed_tokens = sequence.window->removed_tokens();
    if (changed)
        sequence.mtp_draft_count = 0;
    return changed;
}

}
