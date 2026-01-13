# Package: testing

This package provides fake/mock implementations for testing the IPVS proxier without requiring actual Linux IPVS kernel support.

## Key Types

- `FakeIPVS` - Mock implementation of the IPVS interface for unit testing
- Stores virtual servers and real servers in memory maps

## Key Functions

- `NewFake()` - Creates a new fake IPVS interface
- `AddVirtualServer()` - Adds a virtual server to the fake
- `UpdateVirtualServer()` - Updates a virtual server
- `DeleteVirtualServer()` - Removes a virtual server
- `AddRealServer()` - Adds a real server (endpoint) to a virtual server
- `DeleteRealServer()` - Removes a real server
- `GetVirtualServers()` - Returns all virtual servers
- `GetRealServers()` - Returns real servers for a virtual server

## Design Notes

- Enables unit testing of IPVS proxier logic without Linux kernel dependencies
- Maintains in-memory state that mirrors what would be in the kernel
- Validates operations are called with expected parameters
- Used extensively in proxier unit tests
