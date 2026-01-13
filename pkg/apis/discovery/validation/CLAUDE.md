# Package: validation

## Purpose
Provides validation logic for EndpointSlice resources in the discovery.k8s.io API group.

## Key Functions

- **ValidateEndpointSlice(endpointSlice, oldEndpointSlice)**: Full validation including metadata, addressType, ports, endpoints.
- **ValidateEndpointSliceCreate(endpointSlice)**: Validation for new EndpointSlices.
- **ValidateEndpointSliceUpdate(new, old)**: Update validation with immutability check for AddressType.

## Validation Rules

- AddressType: Required, must be IPv4, IPv6, or FQDN.
- Endpoints: Maximum 1000 per slice.
- Addresses: 1-100 per endpoint, validated based on AddressType (IPv4/IPv6/FQDN).
- Ports: Maximum 20000, unique names, protocol required (TCP/UDP/SCTP).
- Port names: Must pass DNS label validation if non-empty.
- NodeName: Must be valid node name.
- Hostname: Must be valid DNS label.
- Zone hints: Maximum 8 per endpoint, valid label values.
- Node hints: Maximum 8 per endpoint, valid node names.
- Topology labels: Maximum 16, reserved keys not allowed.
- AddressType is immutable on update.

## Design Notes

- Preserves invalid Endpoints on update if unchanged (backward compatibility).
- Uses field.ErrorList for error accumulation.
