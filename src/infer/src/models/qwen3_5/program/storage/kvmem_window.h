#pragma once

#include "core/host_kv_arena.h"
#include "models/qwen3_5/program/storage/kv_store.h"
#include "ninfer/types.h"
#include <memory>

namespace ninfer::models::qwen3_5::detail {

struct KvmemBinding {
    LogicalKVPageStore* pages;
    KVAddressSpaceStore* addresses;
    KVAddressSpaceHandle address;
};

class KvmemWindow {
public:
    KvmemWindow(KvmemOptions options, std::uint32_t logical_capacity,
                 const KVPageGeometry& main, const KVPageGeometry* mtp);
    std::uint32_t removed_tokens() const noexcept { return removed_; }
    std::uint32_t compact_position(std::uint32_t logical) const;
    void commit(std::uint32_t frontier);
    bool prepare(std::uint32_t frontier, std::uint32_t append_end, KvmemBinding main,
                 const KvmemBinding* mtp, const std::vector<double>& scores,
                 cudaStream_t stream, bool force = false);
    std::uint64_t swaps() const noexcept { return swaps_; }
    std::uint64_t spilled_bytes() const noexcept { return spilled_; }
    std::uint64_t restored_bytes() const noexcept { return restored_; }

private:
    KvmemOptions options_;
    HostKVPageLayout main_layout_, mtp_layout_;
    std::unique_ptr<HostKVArena> arena_;
    HostKVAllocation main_staging_, mtp_staging_;
    std::vector<std::vector<std::byte>> main_archive_, mtp_archive_;
    std::vector<std::uint32_t> valid_, pages_;
    std::uint32_t removed_ = 0, frontier_ = 0;
    bool invalid_ = false;
    std::uint64_t swaps_ = 0, spilled_ = 0, restored_ = 0;
};

}
