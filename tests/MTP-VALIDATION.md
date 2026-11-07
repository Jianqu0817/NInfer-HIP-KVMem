# MTP state checks

Run from the repository root. Host arithmetic executes the same scalar
continuation function used by the device kernel, including non-KVMem equivalence
and absolute/compact divergence. It does not exercise GPU memory or GDN folding.

```powershell
New-Item -ItemType Directory -Force build/host-tests | Out-Null
g++ -std=c++17 -Wall -Wextra -pedantic -I src/infer/src tests/mtp_round_state_test.cpp -o build/host-tests/mtp_round_state_test.exe
./build/host-tests/mtp_round_state_test.exe
```

With a Windows HIP SDK, VS build environment and the normal engine static library
already built, compile the real-op test using the same flags and include paths
as the engine. Set `$HipSdk` to the SDK root. The engine build does not compile
test mains into its library.

```powershell
& "$HipSdk/bin/hipcc.exe" -std=c++20 --offload-arch=gfx1201 -mno-wavefrontsize64 -I src/shim -I src/infer/include -I src/infer/src tests/mtp_round_gpu_test.cpp build/windows/ninfer_hip.lib -o build/windows/mtp_round_gpu_test.exe
./build/windows/mtp_round_gpu_test.exe
```

The real-op test checks alignment, cache and RoPE positions, logical and physical
tail limits, zero/partial/full draft acceptance, and graph replay after changing
the compact view. Keep `NINFER_MTP_AR_POS_SHIFT` unset: that existing diagnostic
deliberately changes kernel positions.

Observed run on 2026-11-06: the test, `core/tensor.cpp`, `core/dtype.cpp`, the real
MTP wrapper/launcher, and `core/device.cu` compiled and linked using HIP 7.14.1,
MSVC 14.44 and the Windows 26100 SDK. On the RX 9070 XT (`gfx1201`), with
`HIP_VISIBLE_DEVICES=0` and diagnostic overrides unset, the executable printed
`MTP round GPU/graph: PASS` and exited 0. The isolated build commands and run
receipt are in `build/mtp-smoke/`; no stub or CPU substitute was used.

Still required with actual model weights: compare the accepted GDN/conv/tail state
against forced-token execution; cancel and terminal-truncate rounds at partial
page tails; change the dual-pool working set and rebuild draft proposals; verify
repeated query replay does not add LRFU feedback. These are model integration
qualification gates, not covered by the arithmetic or preparation-op tests.
