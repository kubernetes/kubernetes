# Package: stats

## Purpose
The `stats` package handles exporting Kubelet and container statistics via HTTP endpoints. It provides node resource consumption data including CPU, memory, filesystem, and volume metrics.

## Key Types/Structs

- **Provider**: Interface defining methods for retrieving stats from CRI, cAdvisor, and Kubelet (pod stats, cgroup stats, filesystem stats, node info).
- **SummaryProvider**: Interface for generating stats summaries with Get() and GetCPUAndMemoryStats() methods.
- **ResourceAnalyzer**: Combines fsResourceAnalyzer and SummaryProvider for comprehensive resource analysis.
- **fsResourceAnalyzer**: Manages filesystem resource stats with background caching of pod volume stats.
- **volumeStatCalculator**: Calculates volume metrics for a pod periodically in the background.
- **PodVolumeStats**: Contains ephemeral and persistent volume stats for a pod.

## Key Functions

- **CreateHandlers**: Creates REST handlers for stats endpoints (e.g., /stats/summary).
- **NewResourceAnalyzer**: Creates a ResourceAnalyzer with filesystem analysis and summary provider.
- **NewSummaryProvider**: Creates a SummaryProvider for generating stats summaries.
- **handleSummary**: HTTP handler for /stats/summary endpoint, supports "only_cpu_and_memory" parameter.

## HTTP Endpoints

- `/stats/summary` - Returns comprehensive node and pod statistics (supports GET and POST).

## Design Notes

- Designed for loose coupling to support potential extraction into a standalone pod.
- Volume stats are calculated in background goroutines and cached to avoid blocking requests.
- Uses atomic values for thread-safe cache updates.
- Supports platform-specific system container stats (Linux vs Windows).
- The summary includes node stats (CPU, memory, swap, network, filesystem) and per-pod stats.
