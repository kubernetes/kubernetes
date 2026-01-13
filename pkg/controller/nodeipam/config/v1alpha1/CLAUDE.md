# Package: v1alpha1

Versioned API types and defaulting for Node IPAM controller configuration.

## Key Functions

- `RecommendedDefaultNodeIPAMControllerConfiguration()`: Currently empty - mask size defaults depend on cluster CIDR family which is determined at runtime.

## Key Files

- `defaults.go`: Default value functions (placeholder)
- `conversion.go`: Conversion functions between v1alpha1 and internal types
- `register.go`: Scheme registration

## Purpose

Provides the v1alpha1 versioned configuration API for the Node IPAM controller.

## Design Notes

- No static defaults because appropriate mask sizes depend on the IP family
- IPv4 typically uses /24, IPv6 typically uses /64 for node CIDRs
- Defaults are set elsewhere based on cluster configuration
