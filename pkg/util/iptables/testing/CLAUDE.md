# Package: testing

## Purpose
Provides fake implementations of the iptables interface for unit testing networking components without requiring actual iptables.

## Key Types
- `FakeIPTables` - No-op implementation of iptables.Interface
- `IPTablesDump` - In-memory representation of iptables state
- `Table` / `Chain` / `Rule` - Data structures mirroring iptables concepts

## Key Functions
- `NewFake()` - Creates a fake IPv4 iptables
- `NewIPv6Fake()` - Creates a fake IPv6 iptables
- `SetHasRandomFully()` - Configures --random-fully support
- `ParseIPTablesDump()` - Parses iptables-save output format
- `ParseRule()` - Parses individual iptables rules

## Features
- Maintains in-memory state that mirrors real iptables
- Supports SaveInto/Restore operations
- Validates rule syntax and chain references
- Pre-populated with standard chains (INPUT, OUTPUT, FORWARD, etc.)

## Design Patterns
- Implements same interface as production code
- Stateful mock that tracks chains and rules
- Validates restore operations (chain existence, jump targets)
- Useful for testing kube-proxy rule generation
