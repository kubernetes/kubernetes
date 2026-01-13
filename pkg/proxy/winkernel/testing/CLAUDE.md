# Package: testing

This package provides mock implementations of Windows HCN (Host Compute Network) interfaces for unit testing the Windows kube-proxy without requiring actual Windows HNS.

## Key Types

- `HcnMock` - Mock implementation of the HCN service interface

## Key Functions

- `NewHcnMock()` - Creates a new HCN mock with a specified network
- `GetNetworkByName()` - Returns the mock network by name
- `CreateEndpoint()` - Creates a mock HNS endpoint
- `DeleteEndpoint()` - Deletes a mock endpoint
- `CreateLoadBalancer()` - Creates a mock HNS load balancer
- `DeleteLoadBalancer()` - Deletes a mock load balancer
- `ListEndpoints()` - Lists all mock endpoints
- `ListLoadBalancers()` - Lists all mock load balancers

## Design Notes

- Enables unit testing of Windows proxy logic without Windows dependencies
- Maintains in-memory state of endpoints and load balancers
- Simulates HCN API v2 behavior including supported features
- Supports DSR, dual-stack, and remote subnet feature simulation
