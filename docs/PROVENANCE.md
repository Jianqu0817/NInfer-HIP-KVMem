
# 来源与快照范围

- HIP基线：[Newaiguy/NInfer-HIP](https://github.com/Newaiguy/NInfer-HIP)，
  commit `dcbab08e8027fa608778995fecde186536c07860`。
- 原始NInfer：[Neroued/ninfer](https://github.com/Neroued/ninfer)，Apache-2.0；根LICENSE/NOTICE保留。
- donor：[melis2023/ninfer-kvmem-actkv](https://github.com/melis2023/ninfer-kvmem-actkv)，
  commit `a60025a12546cf8a45e084836ae7a77e322fcbf0`。
- `src/infer/src/models/qwen3_5/execution/lrfu_history.h`：从donor
  `kvmem/include/kvmem/lrfu.hpp`提取RAM LRFU state/history，未保留snapshot I/O，算术规则保留。
- `src/infer/src/ops/sparse_moe/decode/sparse_moe_decode_plan.cpp`：恢复donor
  `backends/ninfer/src/ops/sparse_moe/decode/sparse_moe_decode_plan.cpp`，实现未改变。
- HIP KVMem bridge、logical/compact MTP修正、测试和Windows构建入口来自本地未提交改进。

本私人仓库采用独立的初始快照提交；不冒充上游官方发布或上游性能。
`docs/SOURCE-MANIFEST.json`记录本次复制文件的来源hash与导出hash，
以及被排除的备份文件名称。源码按字节复制，唯一导出脚本调整为runner要求显式--sdk。
归档README、文档和ignore规则为新文件；原始开发工作区未改动。

bundled cpp-httplib、ggml-quants、llama-jinja、nlohmann/json、spdlog/fmt、utf8proc
保留各自许可文件。模型未包含；模型条款由其发布者单独规定。
