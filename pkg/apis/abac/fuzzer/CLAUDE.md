# Package: fuzzer

## Purpose
Provides fuzzer functions for the ABAC API group used in property-based testing.

## Key Variables
- `Funcs` - Returns an empty slice of fuzzer functions for the ABAC API group

## Design Notes
- The ABAC types are simple enough that no custom fuzzer functions are needed
- The empty function list means the default fuzzer behavior is used for all ABAC types
- This follows the standard pattern for API group fuzzer packages
