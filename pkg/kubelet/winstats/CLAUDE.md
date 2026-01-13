# Package: winstats

## Purpose
The `winstats` package provides a Windows-specific client to collect node and pod level statistics using Windows Performance Counters and system APIs.

## Key Interfaces

- **Client**: Interface for getting stats information.
  - `WinContainerInfos()`: Returns container info map (analogous to cAdvisor GetContainerInfoV2).
  - `WinMachineInfo()`: Returns machine info (analogous to cAdvisor MachineInfo).
  - `WinVersionInfo()`: Returns version info (kernel, OS).
  - `GetDirFsInfo()`: Returns filesystem capacity and usage information.

## Key Types

- **StatsClient**: Implements the Client interface.
- **perfCounterNodeStatsClient**: Implementation using Windows Performance Counters.
- **nodeMetrics**: CPU usage, memory working set, committed bytes, network stats.
- **nodeInfo**: Physical memory, kernel version, OS image version, start time.
- **MemoryStatusEx**: Windows MEMORYSTATUSEX structure wrapper.
- **PerformanceInformation**: Windows PERFORMANCE_INFORMATION structure wrapper.

## Key Functions

- **NewPerfCounterClient**: Creates a client using Windows Performance Counters.
- **ProcessorCount**: Returns logical processor count across all processor groups.
- **GetPerformanceInfo**: Gets Windows performance information.

## Design Notes

- Windows-only implementation (build tag: windows).
- Uses Windows kernel32.dll and psapi.dll system calls.
- Monitors CPU, memory, and network via performance counters.
- Handles multiple processor groups (supports >64 logical processors).
- Provides cAdvisor-compatible data structures for kubelet integration.
