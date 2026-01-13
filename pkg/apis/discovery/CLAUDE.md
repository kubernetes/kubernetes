# Package: discovery

## Purpose
Internal (unversioned) API types for the discovery.k8s.io API group, providing EndpointSlice resources that represent subsets of service endpoints for scalable endpoint management.

## Key Types

- **EndpointSlice**: Represents a subset of endpoints for a service. Multiple EndpointSlices are joined to produce the full set of endpoints.
- **Endpoint**: Single logical backend with addresses, conditions (ready/serving/terminating), hostname, node name, zone, and hints.
- **EndpointConditions**: Ready, Serving, and Terminating boolean flags.
- **EndpointHints**: Topology-aware routing hints with ForZones and ForNodes.
- **EndpointPort**: Port definition with name, protocol, port number, and appProtocol.
- **AddressType**: IPv4, IPv6, or FQDN (deprecated).

## Key Constants

- **AddressTypeIPv4, AddressTypeIPv6, AddressTypeFQDN**: Supported address types.

## Key Functions

- **Kind(kind string)**: Returns Group-qualified GroupKind.
- **Resource(resource string)**: Returns Group-qualified GroupResource.
- **AddToScheme**: Registers EndpointSlice and EndpointSliceList with a scheme.

## Design Notes

- Maximum 1000 endpoints per slice, 100 addresses per endpoint, 100 ports per slice.
- AddressType is immutable after creation.
- Topology hints enable zone/node-aware routing (TopologyAwareHints feature gate).
- Replaces the legacy Endpoints API for better scalability.
