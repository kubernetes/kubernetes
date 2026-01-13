# Package: servicecidr

## Purpose
Provides utilities for working with ServiceCIDR resources, enabling queries about IP address and prefix containment within cluster service IP ranges.

## Key Functions

### Prefix Queries
- `OverlapsPrefix(serviceCIDRLister, prefix netip.Prefix) []*networkingv1.ServiceCIDR` - Returns all ServiceCIDRs that overlap with the given prefix
- `ContainsPrefix(serviceCIDRLister, prefix netip.Prefix) []*networkingv1.ServiceCIDR` - Returns ServiceCIDRs that fully contain the given prefix (same overlap but with equal or larger mask)

### IP/Address Queries
- `ContainsIP(serviceCIDRLister, ip net.IP) []*networkingv1.ServiceCIDR` - Returns ServiceCIDRs containing the given net.IP
- `ContainsAddress(serviceCIDRLister, address netip.Addr) []*networkingv1.ServiceCIDR` - Returns ServiceCIDRs containing the given netip.Addr

### Helper Functions
- `PrefixContainsIP(prefix netip.Prefix, ip netip.Addr) bool` - Checks if IP is within prefix, excluding network address and (for IPv4) broadcast address
- `IPToAddr(ip net.IP) netip.Addr` - Converts net.IP to netip.Addr, handling IPv4/IPv6 correctly
- `broadcastAddress(subnet netip.Prefix) (netip.Addr, error)` - Calculates the broadcast address for a subnet (internal)

## Design Notes
- PrefixContainsIP excludes network and broadcast addresses since ServiceCIDRs won't allocate those IPs
- Uses netip package for modern IP address handling
- All query functions use a ServiceCIDR lister for efficient lookups
