"""Run one GPU command and retain its evidence; timeouts never kill the process."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "build" / "qualification"


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def receipt_reference(path):
    if path is None:
        return None
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class ProcessMemory:
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]

    def __init__(self, pid):
        self.handle = None
        self.peak = None
        if os.name == "nt":
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            self.kernel.OpenProcess.restype = wintypes.HANDLE
            self.kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            self.read = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            self.read.argtypes = (wintypes.HANDLE, ctypes.POINTER(self.Counters), wintypes.DWORD)
            self.read.restype = wintypes.BOOL
            self.handle = self.kernel.OpenProcess(0x0410, False, pid)

    def sample(self):
        if self.handle:
            counters = self.Counters()
            counters.cb = ctypes.sizeof(counters)
            if self.read(self.handle, ctypes.byref(counters), counters.cb):
                self.peak = max(self.peak or 0, counters.PeakWorkingSetSize)
        return self.peak

    def close(self):
        self.sample()
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sdk", type=Path, required=True, help="HIP SDK root with runtime DLLs")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--input-receipt", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- executable argument ...")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("a command and a positive --timeout are required")
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in args.label) or args.label in (".", ".."):
        parser.error("--label must be a filename using letters, numbers, '-', '_' or '.'")
    references = {"model": receipt_reference(args.model_receipt), "input": receipt_reference(args.input_receipt)}
    QUALIFICATION.mkdir(parents=True, exist_ok=True)
    marker = QUALIFICATION / "gpu-running.json"
    if marker.exists():
        raise RuntimeError(f"A prior GPU job may still be alive; inspect {marker} before starting another")
    case = QUALIFICATION / args.label
    case.mkdir(exist_ok=False)
    with marker.open("x", encoding="utf-8") as file:
        json.dump({"phase": "starting", "label": args.label}, file)
    receipt = case / "receipt.json"
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["PATH"] = str(args.sdk / "bin") + os.pathsep + str(args.sdk / "lib/llvm/bin") + os.pathsep + env.get("PATH", "")
    if os.name == "nt":
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    result = {"label": args.label, "command": command, "cwd": str(ROOT), "sdk": str(args.sdk),
              "HIP_VISIBLE_DEVICES": "0", "local_receipts": references,
              "runtime_env": {name: env.get(name) for name in (
                  "ROCBLAS_USE_HIPBLASLT", "NINFER_KVMEM_TRACE",
                  "NINFER_PROFILE_PREFILL", "NINFER_PROFILE_DECODE", "NINFER_PROFILE_MIXER")},
              "exit_code": None, "rss_peak_bytes": None,
              "stdout": str(case / "stdout.log"), "stderr": str(case / "stderr.log"),
              "started_utc": datetime.now(timezone.utc).isoformat(), "phase": "starting"}
    executable = Path(command[0])
    if not executable.is_absolute():
        executable = ROOT / executable
    if executable.is_file():
        with executable.open("rb") as binary:
            result["executable_sha256"] = hashlib.file_digest(binary, "sha256").hexdigest()
    save(receipt, result)
    started = time.monotonic()
    memory = None
    process = None
    try:
        with (case / "stdout.log").open("wb") as stdout, (case / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            memory = ProcessMemory(process.pid)
            result.update(phase="running", pid=process.pid)
            save(receipt, result)
            save(marker, {"phase": "running", "label": args.label, "pid": process.pid, "receipt": str(receipt)})
            print(json.dumps({"phase": "running", "pid": process.pid, "receipt": str(receipt)}), flush=True)
            deadline = started + args.timeout
            while process.poll() is None:
                result["rss_peak_bytes"] = memory.sample()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result.update(phase="still_running_after_timeout", wall_seconds=time.monotonic() - started,
                                  note="No kill sent; inspect this PID before starting another GPU job")
                    save(receipt, result)
                    save(marker, {"phase": result["phase"], "label": args.label,
                                  "pid": process.pid, "receipt": str(receipt)})
                    print(json.dumps(result), flush=True)
                    return 2
                try:
                    process.wait(timeout=min(1.0, remaining))
                except subprocess.TimeoutExpired:
                    pass
            memory.close()
            result.update(phase="exited", exit_code=process.returncode, wall_seconds=time.monotonic() - started,
                          rss_peak_bytes=memory.peak, rss_peak_source="Windows PSAPI PeakWorkingSetSize" if memory.peak is not None else None)
        save(receipt, result)
        marker.unlink()
        print(json.dumps(result), flush=True)
        return 0 if process.returncode == 0 else 1
    except BaseException as error:
        result.update(phase="error", error=str(error), wall_seconds=time.monotonic() - started)
        if process is not None and process.poll() is None:
            result.update(pid=process.pid, process_still_running=True,
                          note="No kill sent; inspect this PID before starting another GPU job")
            save(marker, {"phase": "error_process_still_running", "pid": process.pid, "receipt": str(receipt)})
        else:
            marker.unlink(missing_ok=True)
        save(receipt, result)
        raise
    finally:
        if memory is not None:
            memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
