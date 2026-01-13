# Package: conntrack

This package provides utilities for managing Linux conntrack (connection tracking) entries, which is essential for cleaning up stale NAT connections when services or endpoints change.

## Key Types

- `Interface` - Main interface for conntrack operations (cleanup, sysctl management)
- `Conntrack` - Real implementation using netlink to interact with kernel conntrack
- `FakeConntrack` - Testing implementation that records operations

## Key Functions

- `NewConntrack()` - Creates a new conntrack interface for a given IP family
- `CleanupServiceConntrack()` - Clears conntrack entries for a deleted service
- `CleanupEndpointsConntrack()` - Clears conntrack entries for changed endpoints
- `CleanupConntrackForPort()` - Clears UDP conntrack entries for a specific port
- `EnsureSysctl()` - Sets required sysctl values (e.g., nf_conntrack_tcp_be_liberal)

## Design Notes

- Uses netlink protocol to communicate with kernel conntrack subsystem
- Supports both IPv4 and IPv6 connection tracking
- UDP connections require special handling due to lack of connection state
- The tcp_be_liberal sysctl helps prevent connection drops during rule updates
- Filter-based cleanup allows precise targeting of stale connections
