# Package: fuzzer

## Purpose
Provides fuzzer functions for the admission API group used in property-based testing.

## Key Variables
- `Funcs` - Returns fuzzer functions for admission types

## Fuzzer Functions
- `runtime.RawExtension` fuzzer - Creates unstructured objects for Object/OldObject fields with apiVersion "unknown.group/unknown" and kind "Something"

## Design Notes
- The RawExtension fuzzer ensures that embedded objects have valid structure for round-trip testing
- Uses unstructured objects to avoid dependency on specific API types during fuzzing
