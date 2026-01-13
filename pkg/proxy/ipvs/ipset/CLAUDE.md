# Package: ipset

This package provides an interface for managing Linux IP sets, which are used by the IPVS proxier for efficient IP address matching in iptables rules.

## Key Types

- `Interface` - Main interface for ipset operations (create, add, delete, list)
- `IPSet` - Represents an IP set with name, type, and hash size configuration
- `Entry` - Represents a single entry in an IP set (IP, port, protocol)
- `Type` - Enum for ipset types (HashIPPort, HashIPPortIP, HashIPPortNet, BitmapPort)

## Key Functions

- `New()` - Creates a new ipset interface
- `CreateSet()` - Creates a new IP set with specified type and options
- `AddEntry()` - Adds an entry to an existing IP set
- `DelEntry()` - Removes an entry from an IP set
- `ListEntries()` - Returns all entries in an IP set
- `FlushSet()` - Removes all entries from a set
- `DestroySet()` - Deletes an IP set entirely

## Design Notes

- Wraps the `ipset` command-line tool for kernel interaction
- Supports various hash types for different use cases (IP:port, IP:port:IP, etc.)
- Used by IPVS proxier for cluster IP, node port, and load balancer IP matching
- Provides atomic operations for rule updates
