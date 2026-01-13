# Package: interrupt

## Purpose
Provides utilities for handling OS termination signals and ensuring cleanup functions run even when a process is interrupted.

## Key Types
- `Handler` - Guarantees execution of notification functions after a critical section or on signal

## Key Functions
- `New()` - Creates a handler with optional final handler and notification functions
- `Run()` - Executes a function with signal handling, running notifications on completion or interrupt
- `Close()` - Manually triggers notification functions
- `Signal()` - Called when OS signal received, runs notifications then final handler

## Handled Signals
- SIGHUP, SIGINT, SIGTERM, SIGQUIT

## Design Patterns
- Guarantees exactly-once execution of notifications via sync.Once
- Default final handler is os.Exit(1)
- Notifications run in order before final handler
- Used for cleanup operations that must run (temp files, locks, etc.)
