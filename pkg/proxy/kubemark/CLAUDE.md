# Package: kubemark

This package provides a hollow (no-op) proxy implementation for kubemark, a tool for simulating large Kubernetes clusters for scalability testing.

## Key Types

- `HollowProxy` - A no-op proxy implementation that simulates kube-proxy without actually programming network rules

## Key Functions

- `NewHollowProxy()` - Creates a new hollow proxy instance
- `Run()` - Starts the hollow proxy (watches Services/EndpointSlices but takes no action)

## Design Notes

- Used in kubemark to simulate kube-proxy at scale without real networking overhead
- Watches Service and EndpointSlice objects to simulate realistic API load
- Does not program any iptables, IPVS, or other network rules
- Useful for testing control plane scalability without node-level overhead
- Implements the same interfaces as real proxy backends
