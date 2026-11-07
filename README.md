
# NInfer-HIP + KVMem — 私人技术储备

原生 Windows、AMD RX 9070 XT / gfx1201 上的 Qwen3.8-27B 文本推理原型。
本快照保留 bounded resident KV + RAM history 的 HIP KVMem 集成、MTP width 2
逻辑／物理位置修正、真实 Engine 文本 CLI 和本地可重建入口。

## 保留的内容

- `src/`：当前引擎源码、HIP shim 和构建所需的 bundled dependencies；许可证随源码保留。
- `harness/`：Windows 构建器、文本 CLI、host 测试入口、串行 GPU 运行收据工具。
- `tests/`：页选择／LRFU／MTP continuation host 检查，native KV 双池和 MTP GPU 检查。
- `docs/`：构建、实现边界、实验摘要和来源信息。

没有打包模型、SDK、二进制、编译对象、历史 `.bak`、大型合成输入或原始实验日志。
这是从本地工作区导出的单次源码快照，未携带上游 Git 历史；原工作区保持不变。

## 快速入口

依赖 Python 3、VS 2022 C++ Build Tools / Windows SDK，以及 Windows HIP SDK。
已使用 ROCm SDK 7.14.1、MSVC 14.44、Windows SDK 26100 构建；本机独显为 RX 9070 XT 16GB。
模型需自行获取 `.ninfer` v3 artifact；本地验证型号为
`WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3`。

```powershell
python harness/windows_build.py --plan
python harness/windows_build.py --check --rocm "$HipSdk" --vcvars "$Vcvars"
python harness/windows_build.py --build --jobs 2 --rocm "$HipSdk" --vcvars "$Vcvars"
```

完整命令见 [构建与运行](docs/BUILD.md)。
[实现边界](docs/IMPLEMENTATION.md) · [实验摘要](docs/VALIDATION.md) · [来源](docs/PROVENANCE.md)

## 状态

真实 131072-token 合成召回输入已在 KVMem＋MTP 同时开启时完成生成并正常退出。
当前 KVMem 关闭 graph，使用 eager；没有因此保证所有权重／工作缓冲均驻留显存。
这是技术储备原型：不支持跨请求 KVMem prefix/session 复用，未交付 HTTP 或交互式聊天服务。
实测吞吐与上游官方 graph 二进制不同，详见实验摘要。

## 来源与许可

基于 [Newaiguy/NInfer-HIP](https://github.com/Newaiguy/NInfer-HIP)，
base commit `dcbab08e8027fa608778995fecde186536c07860`；上游 NInfer 为 Apache-2.0。
LRFU 与一份 sparse-MoE planner 源于
[melis2023/ninfer-kvmem-actkv](https://github.com/melis2023/ninfer-kvmem-actkv)
的固定 commit，具体路径与改动见来源说明。第三方依赖保留各自许可。
