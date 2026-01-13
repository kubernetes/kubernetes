# Package: testing

Testing utilities for the prober package.

## Key Types

- **FakeManager**: Mock implementation of prober.Manager for testing.

## Key Functions

- `AddPod()`: No-op.
- `RemovePod()`: No-op.
- `StopLivenessAndStartup()`: No-op.
- `CleanupPods()`: No-op.
- `Start()`: No-op.
- `UpdatePodStatus()`: Sets all container statuses to Ready=true.

## Design Notes

- Simplest possible implementation for unit testing
- All containers automatically marked as Ready
- No actual probe logic executed
