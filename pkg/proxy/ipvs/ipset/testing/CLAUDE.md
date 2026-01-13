# Package: testing

This package provides a fake implementation of the ipset.Interface for unit testing the IPVS proxier without requiring actual Linux ipset kernel support.

## Key Types

- `FakeIPSet` - Mock implementation of ipset.Interface that stores sets and entries in memory

## Key Functions

- `NewFake()` - Creates a new fake ipset interface
- `CreateSet()` - Records set creation in memory
- `AddEntry()` - Adds entries to the fake set
- `DelEntry()` - Removes entries from the fake set
- `ListEntries()` - Returns entries stored in the fake set
- `FlushSet()` - Clears all entries from a fake set
- `DestroySet()` - Removes a fake set entirely

## Design Notes

- Enables unit testing of IPVS proxier logic without Linux kernel dependencies
- Maintains in-memory state that can be inspected by tests
- Validates that operations are called with expected parameters
- Supports the full ipset.Interface contract
