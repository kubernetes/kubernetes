# Package: async

## Purpose
Provides utilities for managing groups of goroutines that can be started and stopped together.

## Key Types
- `Runner` - Manages a group of loop functions that run until signaled to stop

## Key Functions
- `NewRunner()` - Creates a Runner with one or more loop functions
- `Start()` - Starts all registered loop functions in separate goroutines
- `Stop()` - Signals all running loop functions to stop by closing the stop channel

## Design Patterns
- Each loop function receives a stop channel and should exit when it closes
- Thread-safe start/stop operations with mutex protection
- Idempotent start/stop (safe to call multiple times)
- Simple coordination pattern for background workers
