# Package: pidlimit

## Purpose
The `pidlimit` package provides Linux-specific functions to retrieve process/task limit statistics for the node, including the maximum PID limit and current running process count.

## Key Types/Constants

- **PIDs**: Internal resource name constant for PID resources (`v1.ResourceName("pid")`).

## Key Functions

- **Stats**: Returns `RlimitStats` containing:
  - `MaxPID`: The system-wide task limit (minimum of `kernel.pid_max` and `kernel.threads-max`).
  - `NumOfRunningProcesses`: Current number of running processes/tasks.
  - `Time`: Timestamp of when stats were collected.

## Implementation Details (Linux)

- Reads `/proc/sys/kernel/pid_max` and `/proc/sys/kernel/threads-max` to determine max PIDs.
- Uses the minimum of both values as the effective limit.
- Prefers reading `/proc/loadavg` to get process count (avoids truncation issue with sysinfo(2) for counts > 65538).
- Falls back to `syscall.Sysinfo()` if `/proc/loadavg` is unavailable.

## Design Notes

- Linux-specific implementation uses build tags (`//go:build linux`).
- Addresses Kubernetes issue #107107 regarding truncated process counts from sysinfo(2).
- Used by the Kubelet to report node-level process statistics.
