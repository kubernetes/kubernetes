# Package: metaproxier

This package provides a meta-proxy implementation that wraps two proxy instances (IPv4 and IPv6) to provide dual-stack service support.

## Key Types

- `MetaProxier` - Wraps two proxy.Provider instances, dispatching traffic based on IP family

## Key Functions

- `NewMetaProxier()` - Creates a new dual-stack meta proxier from IPv4 and IPv6 proxiers
- `Sync()` - Triggers sync on both underlying proxiers
- `SyncLoop()` - Runs the sync loop for both proxiers
- `OnServiceAdd/Update/Delete()` - Routes service events to appropriate proxier based on IP family
- `OnEndpointSliceAdd/Update/Delete()` - Routes endpoint events to appropriate proxier

## Design Notes

- Enables dual-stack (IPv4 + IPv6) service support in kube-proxy
- Routes services and endpoints to the correct proxier based on ClusterIP family
- Both proxiers share the same sync timing to maintain consistency
- Used by iptables, IPVS, and nftables backends for dual-stack support
