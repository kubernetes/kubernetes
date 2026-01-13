# Package: fuzzer

## Purpose
Provides fuzzer functions for the discovery API group (EndpointSlice), used in round-trip testing.

## Key Functions

- **Funcs**: Returns fuzzer functions for discovery types.

## Fuzzer Behavior

- Randomizes AddressType to IPv4, IPv6, or FQDN.
- Ensures port names default to empty string if nil.
- Ensures port protocols default to TCP, UDP, or SCTP if nil.

## Design Notes

- Uses randfill library for random data generation.
- Ensures generated objects match defaulting behavior for round-trip tests.
