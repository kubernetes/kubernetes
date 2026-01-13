# Package: fuzzer

## Purpose
Provides fuzzer functions for the flowcontrol API group (FlowSchema, PriorityLevelConfiguration), used in round-trip testing.

## Key Functions

- **Funcs**: Returns fuzzer functions for flowcontrol types.

## Design Notes

- Ensures generated objects have valid distinguisher methods, subject kinds, and priority level types.
- Handles special cases for queuing configuration parameters.
