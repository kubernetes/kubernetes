# Package: runner

This package provides a bounded frequency runner that rate-limits how often a function can be executed, used by kube-proxy to control sync frequency.

## Key Types

- `BoundedFrequencyRunner` - Runs a function with bounded frequency, coalescing multiple requests

## Key Functions

- `NewBoundedFrequencyRunner()` - Creates a runner with min/max interval and burst settings
- `Run()` - Requests a run, which may be deferred based on rate limiting
- `Loop()` - Main loop that waits for run requests and executes the function
- `Stop()` - Stops the runner gracefully

## Design Notes

- Prevents excessive syncs when many service/endpoint changes occur rapidly
- Uses a token bucket algorithm for rate limiting
- Supports burst runs up to a configurable limit
- Minimum interval prevents syncs from happening too frequently
- Maximum interval ensures syncs happen periodically even without changes
- Used by all proxy backends (iptables, IPVS, nftables) for SyncProxyRules
