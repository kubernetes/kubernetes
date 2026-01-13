# Package: testing

This package provides a fake implementation of the IPVS util.Interface for unit testing without requiring Linux IPVS kernel support.

## Key Types

- `FakeIPVS` - Mock implementation of util.Interface that stores state in memory

## Key Functions

- `NewFake()` - Creates a new fake IPVS interface
- `Flush()` - Clears all virtual and real servers
- `AddVirtualServer()` - Adds a virtual server to memory
- `DeleteVirtualServer()` - Removes a virtual server from memory
- `AddRealServer()` - Adds a real server to a virtual server
- `DeleteRealServer()` - Removes a real server

## Design Notes

- Used for unit testing IPVS proxier without kernel dependencies
- Maintains in-memory maps of virtual servers and real servers
- Allows tests to verify expected IPVS state after proxy operations
- Supports the full util.Interface contract
