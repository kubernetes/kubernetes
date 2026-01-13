# Package: fuzzer

## Purpose
Provides fuzz testing functions for the authentication API group types, used for testing serialization roundtrips and API machinery.

## Key Variables
- `Funcs`: A function that returns fuzzer functions for authentication API types

## Current Implementation
The fuzzer currently returns an empty slice, meaning authentication types use default fuzzing behavior without custom fuzzing logic.

## Design Notes
- Authentication types (TokenReview, TokenRequest, SelfSubjectReview) have straightforward structures that work well with default fuzzers
- No special defaulting behavior requires custom fuzzer alignment
