# Package: endpoints

## Purpose
Provides utilities for normalizing and sorting v1.EndpointSubset objects into a canonical form, enabling reliable comparison and consistent API responses.

## Key Functions

### Main Functions
- `RepackSubsets(subsets []v1.EndpointSubset) []v1.EndpointSubset` - Expands endpoint subsets to full representation, then repacks into canonical layout. Returns newly allocated slice.
- `SortSubsets(subsets []v1.EndpointSubset) []v1.EndpointSubset` - Sorts subsets in place by hash, sorts addresses by IP+UID, sorts ports by hash

### Comparison Helpers
- `LessEndpointAddress(a, b *v1.EndpointAddress) bool` - Compares addresses lexicographically by IP, then by TargetRef UID

## Key Types
- `addressKey` - Composite key of IP + UID for deduplicating addresses
- `addressSet` - Map of EndpointAddress pointers to ready state
- `addrReady` - Pairs an address with its ready state for sorting

## Algorithm (RepackSubsets)
1. Map each port to the set of addresses that offer it
2. Map sets of addresses to the ports they collectively offer (using hash as key)
3. Build N-to-M EndpointSubset associations
4. Sort everything for canonical ordering

## Design Notes
- Handles endpoints with no ports using a sentinel value (Port: -1)
- Addresses are deduplicated by IP+UID to handle cases like Mesos where pods share node IP
- Ready state: "not ready" always trumps "ready" for duplicate addresses
- Uses FNV-128a hashing for consistent ordering
