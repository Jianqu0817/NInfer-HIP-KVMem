
# Windows / HIP 构建与运行

## 依赖

- Python 3、VS 2022 Build Tools 的 C++ x64 工具链和 Windows SDK。
- AMD Windows HIP SDK。实际构建使用 ROCm Python SDK 7.14.1 的 unified devel 根目录。
- 不需要旧 `.obj/.lib`；`windows_build.py` 直接枚举仓库内源文件，编译引擎再链接 CLI。
- 当前入口仅文本，不要求 FFmpeg；不把 CUDA-only NVFP4 TMA 声称为可用 HIP kernel。

先按本机安装位置设置变量，再运行：

```powershell
$HipSdk = 'C:\path\to\hip-sdk'
$Vcvars = 'C:\path\to\VS2022\VC\Auxiliary\Build\vcvars64.bat'
$Model = 'C:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer'
python harness/windows_build.py --check --rocm "$HipSdk" --vcvars "$Vcvars"
python harness/windows_build.py --plan
python harness/windows_build.py --build --jobs 2 --rocm "$HipSdk" --vcvars "$Vcvars"
```

成功产物为 `build/windows/ninfer_mini_nordc.exe`。日志和构建清单位于 `build/windows/`，
不进入版本控制。构建器遇到任何编译失败都停止归档／链接，不混用旧对象。

## 短输入运行

```powershell
$env:PATH = "$HipSdk\bin;$HipSdk\lib\llvm\bin;" + $env:PATH
$env:HIP_VISIBLE_DEVICES = '0' # 先确认本机 RX 9070 XT 的枚举号
.\build\windows\ninfer_mini_nordc.exe "$Model" `
  --tasks tests/tasks/sequence_short.tasks --ctx 4096 --max-new 128 `
  --kv rk4v4-e8 --spec mtp --draft-tokens 2 `
  --kvmem --kv-selected 1024 --kv-reserve 2048 --kv-sink 64 --kv-recent 512
```

CLI 使用真实 `Engine::prepare/generate`；默认 thinking=off、temperature=0、seed=0。
`--tasks` 文件为 UTF-8 无 BOM，以 `###ID <name>` 分隔；每个任务是独立请求。
`--ctx` 是逻辑上下文上限，须容纳 prepared 输入＋输出；物理 KV 为 selected＋reserve。
selected/reserve/sink/recent 以64-token页对齐，且 selected＋reserve 不得超过 ctx。

长逻辑输入的已测配置：selected=24576、reserve=4096、sink=256、recent=1024、
prefill-chunk=1024、kv-host-gib=4。输入131072时使用ctx=133120、max-new=256。
仓库不包含长测填充文本；不能用同样 ctx 的短输入代替真实长输入验收。

## ROCm Python SDK 的库目录

本机 unified devel 目录的部分链接曾出现访问错误。必要时只为当前进程指定实际库目录：

```powershell
$LibraryBin = 'C:\path\to\site-packages\_rocm_sdk_libraries\bin'
$env:HIPBLASLT_TENSILE_LIBPATH = "$LibraryBin\hipblaslt\library\gfx1201"
$env:ROCBLAS_TENSILE_LIBPATH = "$LibraryBin\rocblas\library"
```

这些是运行库查找路径，不是关闭计算或替代模型。不要把库加载错误产生的运行当作速度结果。

## 检查与收据

```powershell
python harness/run_host_tests.py --cxx 'C:\path\to\g++.exe' --std c++20
python harness/windows_build.py --link-only --rocm "$HipSdk" --vcvars "$Vcvars" `
  --entry tests/kvmem_native_gpu_test.cpp --name kvmem_native_gpu_test
```

`tests/kvmem_statistics_gpu_test.cpp` 和 `tests/mtp_round_gpu_test.cpp` 可用同一 --entry 方式链接。
GPU 测试及模型推理应串行运行；本次归档没有重新执行 GPU 实验。

```powershell
python harness/run_gpu_case.py --label local-check --sdk "$HipSdk" --timeout 600 `
  -- .\build\windows\ninfer_mini_nordc.exe "$Model" `
  --tasks tests/tasks/sequence_short.tasks --ctx 4096 --max-new 128 --kv rk4v4-e8
```

runner 的 timeout 不会终止 GPU 子进程；若超时，先检查收据中的 PID，勿直接再起一份任务。
本导出版将 --sdk 设为必填，去除了原开发机器的固定 SDK 路径。
