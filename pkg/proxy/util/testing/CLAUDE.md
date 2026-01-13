# Package: testing

This package provides fake implementations of network interfaces for testing proxy utilities without requiring actual network configuration.

## Key Types

- `FakeNetwork` - Mock implementation of NetworkInterfacer for unit testing

## Key Functions

- `NewFakeNetwork()` - Creates a new fake network interface
- `AddInterfaceAddr()` - Adds a network interface with addresses to the fake
- `InterfaceAddrs()` - Returns all addresses across all fake interfaces
- `Interfaces()` - Returns all fake network interfaces
- `InterfaceByName()` - Returns a specific interface by name

## Design Notes

- Enables testing of proxy utilities without real network interfaces
- Allows simulation of various network configurations
- Used to test NodePort address filtering and local traffic detection
- Supports both IPv4 and IPv6 address simulation
