
# KVMem / MTP 实现与边界

## 主要实现

- `KvmemWindow`：native 64-token 页的主模型／MTP 双池、RAM 历史归档与显式恢复。
  归档保留 K/V code 和全部量化 scale planes；最后不足一页的有效列单独维护。
- `memory_statistics`、`LrfuHistory`、`kvmem_policy`：已接受历史的相关性和 LRFU 记录，
  sink/recent 固定页以及其余候选页选择。最终 prefill chunk 提供 query-aware probe。
- KV compact 地址与原逻辑位置分离：cache index 使用物理压缩位置，RoPE/sampling 保持逻辑位置。
- MTP continuation、draft/verify 位置和 tail limit 区分逻辑长度与物理容量；
  页选择变化只发生在 speculative round 已提交之后，拒绝草稿不发布为历史。
- 工作集变化时恢复 GDN/conv/hidden/sampling 状态并重放最终 query chunk。
- `windows_build.py` 和 `mini_main.cpp` 提供仓库内构建及真实 Engine CLI。
- HIP 构建修复包含 GDN 前向声明、CUDA-only NVFP4 TMA 的 unsupported 边界、
  恢复 CMake 引用却缺失的 sparse-MoE planner；IQ3/GDN 计算未替换为 CPU stub。

## 尚未完成

- KVMem 使用 eager，关闭 graph。普通 context-cache/prefix publication 在 KVMem 下禁用。
- 没有跨请求 archive/session reuse；每个独立任务重新处理传入的历史。
- query probe 范围为最后一个 prefill chunk，不是任意长度的完整 user turn。
- 当前文本 CLI 没有 HTTP、--chat 或多模态入口；未验收256K、多GPU或DFlash2。
- KVMem 主动将历史页放入RAM，不代表其他分配都驻留VRAM；Windows Non Local占用需独立观察。
- 现有源码仍含部分无条件诊断输出和计时；未在本次归档中做性能清理。
- scalar／页恢复／GPU preparation 测试不等价于所有真实模型状态路径的完整正确性证明。

## 已有证据

2026-11-06的本机记录包括host规则检查、native rk4v4-e8双池恢复、MTP GPU／graph preparation，
短输入AR/MTP相同输出及一个真实131072-token合成召回fixture。长测事实与时序命中，进程exit 0。
这些证据限定于已执行fixture；原始运行收据留在开发机器，不打包到私人源码仓库。
