# Package: testing

Testing utilities for the pod package, including mock implementations.

## Key Types

- **FakeMirrorClient**: Mock implementation of MirrorClient for testing.

## Key Functions

- `NewFakeMirrorClient()`: Creates a new fake mirror client.
- `CreateMirrorPod()`: Records pod creation in internal set.
- `DeleteMirrorPod()`: Removes pod from internal set.
- `HasPod() / NumOfPods() / GetPods()`: Query internal state for assertions.
- `GetCounts()`: Returns create/delete counts for a specific pod full name.

## Mock Files

- **mocks.go**: Contains mockery-generated mocks for the Manager interface.

## Design Notes

- FakeMirrorClient tracks all created/deleted mirror pods for test assertions
- Uses sets.Set[string] for efficient pod tracking
- Thread-safe with RWMutex protection
- Tracks operation counts per pod for verification
