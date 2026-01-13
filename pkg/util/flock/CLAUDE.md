# Package: flock

## Purpose
Provides file locking utilities for process coordination, ensuring only one instance of a process holds a lock.

## Key Functions
- `Acquire()` - Acquires an exclusive lock on a file for the process lifetime

## Platform Support
- Unix-like systems (Linux, Darwin, FreeBSD, etc.): Uses unix.Flock with LOCK_EX
- Other platforms: May have different or stub implementations

## Design Patterns
- Lock is held until process exits (file descriptor not closed)
- Creates the lock file if it doesn't exist
- Reentrant within the same process
- Used to prevent multiple instances of kubelet or other daemons
