
# 已执行实验摘要

实验日期：2026-11-06。环境：原生Windows 11 build26200，RX9070XT16GB，Ryzen7 7700，
32GB系统内存，ROCm SDK7.14.1。模型：Qwen3.8-27B-GSQ-RCO-IQ3_S，native v3 artifact。
本地模型下载收据SHA256：`29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750`。

## 速度

| 配置／任务 | 实际输入tokens | AR tok/s | MTP tok/s | 范围 |
|---|---:|---:|---:|---|
| 本地KVMem，召回 | 3072 | 8.14 | 17.87 | 各三轮中位数；六份stdout逐字节相同 |
| 本地KVMem，中文讲解 | 91 | — | 13.11 | 生成256tokens；未触发换页 |
| 本地KVMem，代码生成 | 66 | — | 16.47 | 生成256tokens；未触发换页 |
| 本地KVMem，128K简短召回 | 131072 | 未测 | 14.29 | prefill1416.15s，36次换页，生成104tokens |
| 本地KVMem，110K召回＋长说明 | 112640 | 未测 | 11.60 | prefill1214.24s，生成512tokens；30次换页 |
| 本地KVMem，120K | 122880 | — | 未完成 | 用户要求在prefill中停止 |
| 本地KVMem，125K | 128000 | — | 未运行 | 仅校准输入长度；关闭MTP的长输入对照未运行 |
| 官方pre-r7，自己的召回任务，ctx4096 | 3072 | 约9.9 | 21.38–21.42 | 原版graph；无KVMem |
| 官方pre-r7，上游tasks_t3kL，ctx8192 | 2845 | 9.90 | 22.36–22.60 | 各两轮；max-new256 |

本地KVMem版速度为 `(generated-1)/decode_seconds`，首token在prefill产生；不含模型加载。
官方二进制按其CLI直接报告的decode tok/s记录。不同输出任务不能直接归因比较。
KVMem版为eager／graph关闭；原版为默认graph。MTP均draft-tokens2。
120K曾有14–15tok/s的聊天估算，它不是实验结果，不纳入本表。

## 显存观察

110K长说明的生成期21个采样点中，目标进程Local中位数约9.776GiB，Non Local约3.348GiB。
首次KVMem换页之前已观察到约3.276GiB Non Local占用；不能据此宣称128K之前完全没有系统内存占用。
Non Local非零不单独证明是哪一个buffer被驱逐；本轮未采集逐分配迁移事件。
KVMem spilled/restored为主动传输累计字节，不等于Windows Non Local的瞬时驻留量。

128K简短召回完成并命中预期称呼、最终约定和事件顺序，MTP草稿接受率93.1%。
这是一份中性重复文本填充的合成fixture，不代表所有128K文档的召回质量。
该速度轮未开prefill/decode/mixer profiler，仅启用KVMem页trace；源码仍含部分无条件诊断。

## 版本标识

- 本地KVMem测试exe SHA256：`463f01d27e51c56eeea9bf17a9edeb347a706bc83e124f3fb258afe72f6eb00b`。
- 官方pre-r7 exe SHA256：`f1de74ae1c279be5420707127fe86b38dc07f6d4b166eb0b14f470998ccd9a3c`。
- 官方发布测试的HIP环境是7.2.3，本机运行库为7.14.1；没有复现作者发布页的速度。
- 所有正式GPU任务串行。未关闭其他桌面程序；不推断速度差距的唯一原因。

## 归档时检查

本次归档不重跑GPU模型测试；只检查源码清单与hash、构建计划、工具链可发现性和host规则测试。
检查结果见 `docs/ARCHIVE-CHECKS.json`。原始日志、遥测和大型输入均保留在开发机器。
