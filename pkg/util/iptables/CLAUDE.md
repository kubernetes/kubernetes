# Package: iptables

## Purpose
Provides an interface and implementations for running iptables commands, used by kube-proxy and other networking components.

## Key Types
- `Interface` - Main interface for iptables operations
- `runner` - Implementation that executes actual iptables commands
- `Table` - Represents iptables tables (nat, filter, mangle)
- `Chain` - Represents iptables chains (PREROUTING, POSTROUTING, etc.)
- `Protocol` - IPv4 or IPv6

## Key Functions
- `New()` - Creates an iptables interface for a protocol
- `NewBestEffort()` - Creates interfaces for available IP families
- `EnsureChain()` / `DeleteChain()` - Chain management
- `EnsureRule()` / `DeleteRule()` - Rule management
- `SaveInto()` / `Restore()` / `RestoreAll()` - Bulk operations via iptables-save/restore
- `Monitor()` - Detects external iptables flushes via canary chains

## Key Features
- Thread-safe operations with mutex
- Automatic version detection for feature support
- Wait flag support for lock contention
- Both legacy and nft backend support
- Parse error extraction from iptables-restore output

## Design Patterns
- Interface-based for testability
- Version-aware feature detection
- Lock acquisition for iptables-restore on older versions
- Canary chain pattern for detecting external modifications
