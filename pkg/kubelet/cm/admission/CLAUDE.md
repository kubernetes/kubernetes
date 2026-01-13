# Package admission

Package admission provides error handling utilities for pod admission in the container manager context.

## Key Types

- `Error`: Interface for typed admission errors with Error() and Type() methods
- `unexpectedAdmissionError`: Wrapper for unexpected errors during allocation

## Key Functions

- `GetPodAdmitResult(err error) lifecycle.PodAdmitResult`: Converts an error to a PodAdmitResult
  - Returns Admit: true if err is nil
  - Wraps non-admission errors in unexpectedAdmissionError
  - Extracts message and reason from admission errors

## Constants

- `ErrorReasonUnexpected`: "UnexpectedAdmissionError" - reason for unexpected errors

## Design Notes

- Uses errors.As for type-safe error unwrapping
- Allows resource managers to define custom admission error types
- Provides consistent error handling across CPU, memory, device managers
