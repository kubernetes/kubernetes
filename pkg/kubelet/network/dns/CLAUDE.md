# Package: dns

Configures DNS resolver settings for pod containers.

## Key Types

- **Configurer**: Builds DNS configuration for pods based on DNSPolicy.
- **podDNSType**: Internal enum for DNS modes (podDNSCluster, podDNSHost, podDNSNone).

## Key Functions

- `NewConfigurer()`: Creates DNS configurer with cluster DNS settings.
- `GetPodDNS()`: Returns DNSConfig for a pod based on its DNSPolicy.
- `SetupDNSinContainerizedMounter()`: Configures DNS for containerized mount utilities.

## DNS Policies

- **ClusterFirst**: Use cluster DNS, fall back to host for non-cluster domains.
- **ClusterFirstWithHostNet**: ClusterFirst for host network pods.
- **Default**: Use node's DNS settings.
- **None**: Use only pod's dnsConfig fields.

## DNS Configuration

For ClusterFirst pods:
- Nameservers: clusterDNS IPs
- Search domains: `<namespace>.svc.<cluster-domain>`, `svc.<cluster-domain>`, `<cluster-domain>`, plus host domains
- Default options: `ndots:5`

## File Handling

- Reads host DNS from ResolverConfig (typically /etc/resolv.conf)
- Max resolv.conf size: 10MB
- Validates nameservers, search domains, and options

## Design Notes

- Merges cluster DNS with pod-specific dnsConfig overrides
- Deduplicates nameservers and search domains
- Limits search domains and options per DNS spec limits
- Logs warnings for truncated or invalid configurations
