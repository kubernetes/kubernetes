# Package: fuzzer

## Purpose
Provides fuzzer functions for the scheduling API group to generate valid random test data for fuzz testing.

## Key Functions

### Funcs
Returns fuzzer functions for scheduling types:

- **PriorityClass fuzzer**: Ensures PreemptionPolicy is set to PreemptLowerPriority if nil (matching the defaulter behavior)

## Notes
- Simple fuzzer that only handles PreemptionPolicy defaulting
- PreemptLowerPriority is the standard default allowing preemption of lower-priority pods
- Critical for testing scheduler stability with random PriorityClass inputs
