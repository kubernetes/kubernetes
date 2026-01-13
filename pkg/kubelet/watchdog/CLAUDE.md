# Package: watchdog

## Purpose
The `watchdog` package provides a systemd watchdog-based health checker that notifies systemd of kubelet health status.

## Key Types

- **Watchdog**: Interface for health checking.
  - `Start()`: Starts the watchdog loop.
  - `Update()`: Updates health status.

- **healthChecker**: Interface for checking component health.
  - `CheckHealth()`: Returns health status.

## Key Functions

- **NewWatchdog**: Creates a new watchdog instance (Linux only, returns nil on other platforms).

## Behavior (Linux)

1. Reads `WATCHDOG_USEC` environment variable for systemd watchdog timeout.
2. Periodically checks health using the provided health checker.
3. If healthy, notifies systemd via `sd_notify("WATCHDOG=1")`.
4. If unhealthy, stops notifications causing systemd to restart kubelet.
5. Notification interval is half the watchdog timeout for safety margin.

## Platform Support

- **Linux**: Full implementation using systemd notifications.
- **Windows/Others**: Returns nil (no-op).

## Design Notes

- Integrates with systemd's service watchdog feature.
- Health check frequency is derived from systemd timeout.
- Requires kubelet to be managed by systemd with WatchdogSec configured.
- Uses coreos/go-systemd for sd_notify communication.
- Gracefully handles missing watchdog configuration.
