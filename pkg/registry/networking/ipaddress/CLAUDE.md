# Package: ipaddress

Implements the API server registry strategy for IPAddress resources used for IP address management.

## Key Types

- **ipAddressStrategy**: Implements create/update/delete strategies for IPAddress objects.
- **noopNameGenerator**: Custom name generator that returns the base name unchanged.

## Key Functions

- **PrepareForCreate / PrepareForUpdate**: Minimal preparation (no special handling needed).
- **Validate / ValidateUpdate**: Validates IPAddress objects using networking validation.

## Design Notes

- IPAddress is a cluster-scoped resource for tracking IP address allocations.
- Uses a no-op name generator since IPAddress names are typically the IP addresses themselves.
- Part of the Service IP allocation system, used to track which IPs are allocated to Services.
- No status subresource (spec-only resource).
