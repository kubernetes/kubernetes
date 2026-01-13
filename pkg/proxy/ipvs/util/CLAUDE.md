# Package: util

This package provides the interface and implementation for interacting with Linux IPVS (IP Virtual Server) kernel subsystem via netlink.

## Key Types

- `Interface` - Main interface defining IPVS operations
- `VirtualServer` - Represents an IPVS virtual server (service VIP)
- `RealServer` - Represents an IPVS real server (backend endpoint)
- `runner` - Real implementation using netlink to communicate with kernel

## Key Functions

- `New()` - Creates a new IPVS interface using netlink
- `AddVirtualServer()` - Creates a new virtual server in IPVS
- `UpdateVirtualServer()` - Updates virtual server configuration
- `DeleteVirtualServer()` - Removes a virtual server
- `AddRealServer()` - Adds a backend endpoint to a virtual server
- `DeleteRealServer()` - Removes a backend endpoint
- `GetVirtualServers()` - Lists all virtual servers
- `GetRealServers()` - Lists real servers for a virtual server

## Design Notes

- Uses netlink socket to communicate with kernel IPVS module
- Supports various scheduling algorithms (rr, wrr, lc, wlc, etc.)
- Handles both TCP and UDP protocols
- Supports session persistence (sticky sessions)
