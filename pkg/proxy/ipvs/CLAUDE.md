# Package: ipvs

This package implements the IPVS-based kube-proxy backend, which uses Linux IPVS (IP Virtual Server) for high-performance service load balancing.

## Key Types

- `Proxier` - Main IPVS proxy implementation, implements proxy.Provider interface

## Key Functions

- `NewProxier()` - Creates a new IPVS-based proxy instance
- `NewDualStackProxier()` - Creates a dual-stack proxy using metaproxier
- `SyncProxyRules()` - Syncs IPVS virtual servers and real servers with desired state

## Design Notes

- IPVS provides better performance than iptables for large numbers of services
- Uses dummy network interface (kube-ipvs0) to bind service ClusterIPs
- Still uses iptables for masquerading and some filtering rules
- Supports multiple scheduling algorithms (rr, lc, dh, sh, sed, nq)
- Uses ipset for efficient IP matching in iptables rules
- Requires IPVS kernel modules (ip_vs, ip_vs_rr, etc.)
- Real servers represent individual endpoint pods
