# Package: logs

Manages container log rotation and cleanup.

## Key Types/Structs

- **ContainerLogManager**: Interface for managing container log lifecycle (start, clean).
- **LogRotatePolicy**: Configuration for log rotation (MaxSize in bytes, MaxFiles count).
- **containerLogManager**: Implementation that rotates logs when they exceed MaxSize.

## Key Functions

- `NewContainerLogManager()`: Creates a log manager with rotation policy. Returns stub if MaxSize < 0.
- `Start()`: Starts worker goroutines for log rotation and periodic monitoring.
- `Clean()`: Removes all logs for a container (including rotated files).
- `GetAllLogs()`: Returns all log files (rotated/compressed) for a container, sorted oldest to newest.

## Internal Operations

- `rotateLogs()`: Queues running containers for rotation check.
- `rotateLog()`: Performs rotation: cleanup unused, remove excess, compress old, rotate current.
- `compressLog()`: Compresses log files with gzip (.gz suffix).
- `rotateLatestLog()`: Renames current log with timestamp, tells CRI to reopen.
- `removeExcessLogs()`: Keeps only MaxFiles-2 rotated files.

## Design Notes

- Uses workqueue with rate limiting for parallel rotation processing
- Rotated logs named with timestamp suffix (e.g., `log.20060102-150405`)
- Compressed logs have `.gz` suffix
- Temporary files have `.tmp` suffix and are cleaned up
- Thread-safe with mutex protection
- Integrates with CRI RuntimeService for container status and log reopen
