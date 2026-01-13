# Package: exec

## Purpose
The `exec` package implements command execution probes for container health checks.

## Key Interfaces

- **Prober**: Interface for exec probes.
  - `Probe(cmd)`: Executes command and returns result.

## Key Functions

- **New**: Creates a new exec Prober.

## Behavior

1. Executes the provided command.
2. Captures stdout and stderr (limited to 10KB).
3. Returns Success if exit code is 0.
4. Returns Failure if exit code is non-zero.
5. Returns Unknown for other errors.
6. Handles command timeout (returns Failure).

## Constants

- **maxReadLength**: 10KB - maximum output captured from command.

## Design Notes

- Uses LimitWriter to prevent memory exhaustion.
- Combines stdout and stderr into single output.
- Exit code 0 = Success, non-zero = Failure.
- Timeout errors are converted to Failure (not Unknown).
- Used by kubelet for exec-based liveness/readiness probes.
