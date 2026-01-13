# Package: systemd

Provides systemd/logind D-Bus integration for graceful node shutdown on Linux.

## Key Types/Structs

- **DBusCon**: D-Bus connection wrapper with methods for systemd/logind interaction.
- **InhibitLock**: File descriptor handle for systemd shutdown inhibitor lock.

## Key Functions

- `NewDBusCon()`: Creates D-Bus system bus connection.
- `CurrentInhibitDelay()`: Returns current InhibitDelayMaxUSec from logind configuration.
- `InhibitShutdown()`: Creates a "delay" mode inhibitor lock to pause shutdown while kubelet terminates pods.
- `ReleaseInhibitLock()`: Closes the inhibitor lock file descriptor, allowing shutdown to proceed.
- `MonitorShutdown()`: Watches for "PrepareForShutdown" D-Bus signals from logind.
- `ReloadLogindConf()`: Sends SIGHUP to systemd-logind to reload configuration.
- `OverrideInhibitDelay()`: Writes `/etc/systemd/logind.conf.d/99-kubelet.conf` to set InhibitDelayMaxSec.

## D-Bus Services Used

- `org.freedesktop.login1` (logind): Shutdown inhibition and monitoring
- `org.freedesktop.systemd1`: Service management (for config reload)

## Design Notes

- Uses godbus/dbus library for D-Bus communication
- Inhibitor lock is "delay" mode (not "block") - postpones but doesn't prevent shutdown
- InhibitDelayMaxUSec is in microseconds, InhibitDelayMaxSec config is in seconds
- Config override stored in logind.conf.d drop-in directory
- Only available on Linux (build-tagged)
