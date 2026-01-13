# Package: oom

## Purpose
Provides utilities for managing Linux OOM (Out of Memory) killer scores for processes and containers.

## Key Types
- `OOMAdjuster` - Struct with functions for applying OOM score adjustments

## Key Functions
- `NewOOMAdjuster()` - Creates an OOMAdjuster (Linux only)
- `ApplyOOMScoreAdj()` - Sets oom_score_adj for a single process
- `ApplyOOMScoreAdjContainer()` - Sets oom_score_adj for all processes in a cgroup

## Implementation Details
- Writes to /proc/<pid>/oom_score_adj
- PID 0 means "self" (current process)
- Retries on transient errors
- Container-level adjustment iterates until process list stabilizes

## Platform Support
- Linux: Full implementation
- Other platforms: Stub/fake implementations

## Design Patterns
- Dependency injection via function fields for testing
- Handles race conditions with forking processes
- Used by kubelet to protect critical system processes
