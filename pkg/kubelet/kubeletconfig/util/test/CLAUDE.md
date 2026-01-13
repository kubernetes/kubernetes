# Package: test

Testing utilities for kubeletconfig tests.

## Key Functions

- **ExpectError(t, err, substr)**: Asserts that an error contains the expected substring. If substr is empty, expects nil error. Calls t.Fatalf on mismatch.

- **SkipRest(t, desc, err, contains)**: Checks if there was an unexpected error or an expected error that didn't occur. Returns true if the rest of the test case should be skipped. Logs errors via t.Errorf rather than t.Fatalf.

## Usage Notes

- ExpectError is useful for subtests where you want to fail immediately
- SkipRest is useful when you want to continue checking other test cases after logging an error
- Both functions handle the case where an error is expected vs unexpected
