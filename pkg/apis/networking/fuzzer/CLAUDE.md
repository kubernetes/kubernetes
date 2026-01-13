# Package: fuzzer

## Purpose
Provides custom fuzzer functions for the networking API group to generate valid random test data for fuzz testing.

## Key Functions

### Funcs
Returns fuzzer functions for networking types:

- **NetworkPolicyPeer fuzzer**: Generates valid IPBlock with CIDR "192.168.1.0/24" and except list when IPBlock is present
- **NetworkPolicy fuzzer**: Ensures PolicyTypes defaults to ["Ingress"] if empty
- **HTTPIngressPath fuzzer**: Randomly selects PathType from Exact, Prefix, or ImplementationSpecific
- **ServiceBackendPort fuzzer**: Ensures mutual exclusivity between Name and Number fields
- **IngressClass fuzzer**: Defaults Parameters.Scope to "Cluster" if not set
- **IPAddress fuzzer**: Generates random valid IPv4 or IPv6 addresses as object names
- **ServiceCIDR fuzzer**: Generates 1-2 random CIDRs (one per IP family)

### Helper Functions
- `generateRandomIP(is6 bool, c randfill.Continue)`: Creates random IPv4 (4 bytes) or IPv6 (16 bytes) addresses
- `generateRandomCIDR(is6 bool, c randfill.Continue)`: Creates random CIDR notation with random prefix length

## Notes
- Uses `netip` package for IP address parsing and validation
- Ensures generated data meets API validation requirements
- Critical for testing API server stability with random inputs
