# Package: fuzzer

## Purpose
Provides fuzz testing functions for the authorization API group types, used for testing serialization roundtrips and API machinery.

## Key Variables
- `Funcs`: A function that returns fuzzer functions for authorization API types

## Current Implementation
The fuzzer currently returns an empty slice, meaning authorization types use default fuzzing behavior without custom fuzzing logic.

## Design Notes
- Authorization types (SubjectAccessReview, SelfSubjectAccessReview, etc.) have straightforward structures that work well with default fuzzers
- No special defaulting behavior requires custom fuzzer alignment
