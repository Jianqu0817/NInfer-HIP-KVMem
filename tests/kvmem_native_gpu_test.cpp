// Requires a real HIP device and the normal engine archive. No model weights are needed.
#include "core/device.h"
#include "core/host_kv_arena.h"
#include "models/qwen3_5/program/storage/kvmem_window.h"
#include "models/qwen3_5/state/decoder_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::models::qwen3_5;
using namespace ninfer::models::qwen3_5::detail;
constexpr std::uint32_t kFrontier = 3 * 64 + 37;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void check(cudaError_t result) {
    if (result != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(result)); }
}

struct Endpoint {
    PagedKVCache cache;
    LogicalKVPageStore pages;
    KVAddressSpaceStore addresses;
    KVAddressSpaceHandle address;

    Endpoint(DeviceSpan backing, const PagedKVCacheLayout& layout)
        : cache(backing, layout), pages(cache.page_pool(), 4),
          addresses(pages, cache.execution_tables(), 1, 4) {
        auto active = addresses.create_active(4, 0);
        require(active.has_value(), "cannot create native test address");
        address = *active;
        addresses.ensure_mapped_to_tokens(address, kFrontier);
        addresses.commit_frontier(address, kFrontier);
    }

    KvmemBinding binding() { return {&pages, &addresses, address}; }

    std::array<std::int32_t, 4> physical_indices() {
        const auto& row = addresses.execution_row(address);
        const Tensor table = cache.execution_tables().row(row.handle());
        std::array<std::int32_t, 4> indices{};
        check(cudaMemcpy(indices.data(), table.data, sizeof(indices), cudaMemcpyDeviceToHost));
        return indices;
    }
};

std::byte pattern(std::size_t plane, std::uint32_t logical, std::uint32_t head,
                  std::uint32_t token, std::size_t byte, unsigned tag) {
    // The last page has 37 committed columns. Its uncommitted suffix is a distinct sentinel;
    // native page copying must preserve that suffix too, without dequantizing/repacking it.
    if (logical == 3 && token >= 37) { return std::byte(0xD0U + tag); }
    return std::byte((plane * 23 + logical * 47 + head * 11 + token * 7 + byte * 3 + tag) & 255U);
}

void seed_device_directly(Endpoint& endpoint, unsigned tag) {
    const auto mapping = endpoint.physical_indices();
    const auto& geometry = endpoint.cache.page_pool().geometry();
    require(geometry.device_plane_order == PagedKVPlaneOrder::PageMajor,
            "native test expects the target cache's PageMajor layout");
    for (std::size_t p = 0; p < endpoint.cache.page_pool().plane_count(); ++p) {
        const Tensor plane = endpoint.cache.page_pool().plane(p);
        const auto row_bytes = std::size_t(geometry.planes[p].leading_extent) * dtype_size(plane.dtype);
        std::vector<std::byte> bytes(plane.bytes(), std::byte{0xEE});
        for (std::uint32_t logical = 0; logical < 4; ++logical) {
            require(mapping[logical] >= 0 && mapping[logical] < 4, "invalid published physical index");
            for (std::uint32_t head = 0; head < std::uint32_t(geometry.planes[p].head_extent); ++head) {
                for (std::uint32_t token = 0; token < 64; ++token) {
                    const auto offset = std::size_t(mapping[logical]) * plane.nb[3] +
                                        std::size_t(head) * plane.nb[2] + token * plane.nb[1];
                    for (std::size_t b = 0; b < row_bytes; ++b) {
                        bytes[offset + b] = pattern(p, logical, head, token, b, tag);
                    }
                }
            }
        }
        // Seed with raw device-plane addressing, independent of the archive copy helpers.
        check(cudaMemcpy(plane.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    }
}

void verify_native_pages(Endpoint& endpoint, std::array<std::uint32_t, 3> originals,
                         unsigned tag, cudaStream_t stream) {
    auto& pool = endpoint.cache.page_pool();
    const auto layout = plan_host_kv_page_layout(pool.geometry());
    const std::array layouts{layout};
    HostKVArena readback(layout.page_stride * 3, layouts);
    auto allocation = readback.allocate(layout, 3);
    require(allocation.has_value(), "cannot allocate native readback");
    std::array<DeviceKVPageHandle, 3> handles;
    for (std::uint32_t i = 0; i < handles.size(); ++i) {
        handles[i] = endpoint.addresses.physical_page(endpoint.address, i);
    }
    pool.copy_to_host(handles, readback.writable_view(*allocation), stream);
    check(cudaStreamSynchronize(stream));
    const auto data = readback.view(*allocation).data();
    for (std::size_t slot = 0; slot < originals.size(); ++slot) {
        for (std::size_t p = 0; p < layout.planes.size(); ++p) {
            const auto& plane = layout.geometry.planes[p];
            const auto row_bytes = std::size_t(plane.leading_extent) * dtype_size(plane.dtype);
            for (std::uint32_t head = 0; head < std::uint32_t(plane.head_extent); ++head) {
                for (std::uint32_t token = 0; token < 64; ++token) {
                    const auto offset = slot * layout.page_stride + layout.planes[p].offset +
                                        head * layout.planes[p].head_payload_bytes + token * row_bytes;
                    for (std::size_t b = 0; b < row_bytes; ++b) {
                        if (data[offset + b] != pattern(p, originals[slot], head, token, b, tag)) {
                            throw std::runtime_error("native byte mismatch: pool=" + std::to_string(tag) +
                                " slot=" + std::to_string(slot) + " plane=" + std::to_string(p) +
                                " token=" + std::to_string(token));
                        }
                    }
                }
            }
        }
    }
    require(endpoint.addresses.committed_frontier(endpoint.address) == 2 * 64 + 37,
            "compact committed frontier lost the partial tail");
    require(endpoint.pages.committed_columns(endpoint.addresses.logical_page(endpoint.address, 2)) == 37,
            "native tail coverage is not 37 columns");
}
} // namespace

int main() {
    try {
        int devices = 0;
        check(cudaGetDeviceCount(&devices));
        require(devices > 0, "native GPU test requires a real HIP device");
        check(cudaSetDevice(0));
        LayoutBuilder builder;
        const auto layout = plan_decoder_state(builder, DecoderStateSpec{
            .full_attention_layers = 2, .mtp_layers = 1, .capacity = 256,
            .kv_heads = 2, .attention_head_dim = 256,
            .kv_storage = KvCacheStorage::RotatedInt4KeyInt4ValueE8,
            .enable_mtp = true, .kv_table_rows = 1,
            .text_physical_page_groups = 4, .mtp_physical_page_groups = 4,
        });
        require(layout.mtp_kv.has_value(), "native test requires MTP cache");
        require(layout.mtp_kv->layer_storage.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8,
                "unset NINFER_MTP_KV_DTYPE override for the rk4v4-e8 native test");
        DeviceBuffer backing(builder.finish(256));
        const DeviceSpan span{backing.p, backing.bytes};
        Endpoint main(span, layout.text_kv);
        Endpoint mtp(span, *layout.mtp_kv);
        require(main.cache.page_pool().plane_count() == 8 && mtp.cache.page_pool().plane_count() == 4,
                "expected all native K/V/key-scale/value-scale planes");
        seed_device_directly(main, 1);
        seed_device_directly(mtp, 2);
        KvmemOptions options;
        options.enabled = true;
        options.selected_tokens = 192;
        options.reserve_tokens = 64;
        options.sink_tokens = 64;
        options.recent_tokens = 64;
        options.host_capacity_bytes = 16ULL << 20;
        KvmemWindow window(options, 512, main.cache.page_pool().geometry(),
                             &mtp.cache.page_pool().geometry());
        window.commit(kFrontier);
        auto mtp_binding = mtp.binding();
        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream));
        require(window.prepare(kFrontier, kFrontier + 1, main.binding(), &mtp_binding,
                               {0, 10, 1, 0}, stream, true), "first native swap did not run");
        verify_native_pages(main, {0, 1, 3}, 1, stream);
        verify_native_pages(mtp, {0, 1, 3}, 2, stream);
        require(window.prepare(kFrontier, kFrontier + 1, main.binding(), &mtp_binding,
                               {0, 1, 10, 0}, stream, true), "second native swap did not run");
        verify_native_pages(main, {0, 2, 3}, 1, stream);
        verify_native_pages(mtp, {0, 2, 3}, 2, stream);
        require(window.swaps() == 2 && window.removed_tokens() == 64,
                "native swap accounting or original/compact position mapping is wrong");
        check(cudaStreamDestroy(stream));
        std::puts("KVMem native rk4v4-e8 dual-pool restore: PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMem native GPU test failed: %s\n", error.what());
        return 1;
    }
}
