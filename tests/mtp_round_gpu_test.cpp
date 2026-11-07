#include "ninfer/ops/mtp_round.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "HIP/CUDA: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

struct DeviceInts {
    ninfer::Tensor tensor;
    DeviceInts(std::initializer_list<std::int32_t> shape) {
        void* data = nullptr;
        std::size_t elements = 1;
        for (auto n : shape) { elements *= static_cast<std::size_t>(n); }
        check(cudaMalloc(&data, elements * sizeof(std::int32_t)));
        tensor = ninfer::Tensor(data, ninfer::DType::I32, shape);
    }
    ~DeviceInts() { cudaFree(tensor.data); }
    void set(std::initializer_list<std::int32_t> values) {
        check(cudaMemcpy(tensor.data, values.begin(), values.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    }
    void expect(std::initializer_list<std::int32_t> values, const char* name) {
        std::vector<std::int32_t> actual(values.size());
        check(cudaMemcpy(actual.data(), tensor.data, actual.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
        std::size_t i = 0;
        for (auto wanted : values) {
            if (actual[i] != wanted) {
                std::fprintf(stderr, "%s[%zu]: got %d, expected %d\n", name, i, actual[i], wanted);
                std::exit(1);
            }
            ++i;
        }
    }
};

int main() {
    // Call the real wrapper and kernel. The first three rows cover zero,
    // partial and full acceptance; row 3 exercises logical capacity exhaustion.
    constexpr int B = 4;
    constexpr int K = 2;
    DeviceInts ids({K + 1, B}), anchors({B}), accepted({B}), cache_base({B});
    DeviceInts logical_after({B}), budgets({B}), licensed({B}), rope_delta({B});
    DeviceInts alignment({K + 1, B}), extent({B}), positions({B, K - 1});
    DeviceInts rope({B, K - 1}), valid({B, K - 1});
    ids.set({10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42});
    anchors.set({90, 91, 92, 93});
    accepted.set({0, 1, 2, 1});
    cache_base.set({64, 64, 64, 64});
    logical_after.set({100001, 100002, 100003, 131071});
    budgets.set({20, 20, 20, 20});
    licensed.set({1, 2, 3, 2});
    rope_delta.set({99936, 99936, 99936, 131005});
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream));
    const auto launch = [&] {
        ninfer::ops::mtp_prepare_next_round(
            ids.tensor, anchors.tensor, accepted.tensor, cache_base.tensor,
            logical_after.tensor, budgets.tensor, licensed.tensor, rope_delta.tensor,
            alignment.tensor, extent.tensor, positions.tensor, rope.tensor, valid.tensor,
            131072, 128, stream);
    };
    launch();
    check(cudaStreamSynchronize(stream));
    alignment.expect({90, 90, 90, 21, 91, 91, 31, 32, 92, 41, 93, 93}, "alignment");
    positions.expect({65, 66, 67, 66}, "AR cache frontier");
    rope.expect({100001, 100002, 100003, 131071}, "AR logical RoPE");
    extent.expect({2, 2, 2, 0}, "next extent");
    valid.expect({1, 1, 1, 0}, "AR validity");

    // A replayed graph must consume the new view's cache mapping, not capture-time
    // host values. The operation captures only stable tensor backing addresses.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    cache_base.set({125, 124, 123, 64});
    rope_delta.set({99875, 99876, 99877, 131005});
    check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    positions.expect({126, 126, 126, 66}, "graph remapped AR frontier");
    rope.expect({100001, 100002, 100003, 131071}, "graph original RoPE");
    extent.expect({1, 1, 1, 0}, "graph physical capacity");
    valid.expect({0, 0, 0, 0}, "graph invalid AR tail");
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));
    std::puts("MTP round GPU/graph: PASS");
}
