# Package: rlimit

## Purpose
Provides utilities for setting Linux resource limits (rlimits).

## Key Functions
- `SetNumFiles()` - Sets the maximum number of open file descriptors (RLIMIT_NOFILE)

## Implementation Details
- Uses unix.Setrlimit to set both soft and hard limits
- Sets both Cur (soft) and Max (hard) to the same value

## Platform Support
- Linux: Full implementation
- Other platforms: Stub implementations or build-excluded

## Design Patterns
- Simple wrapper around system calls
- Used by components that need many file descriptors (kubelet, API server)
- Should be called early in process startup before limits matter
