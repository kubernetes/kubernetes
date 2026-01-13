# Package: filesystem

## Purpose
Provides filesystem abstractions and utilities for mocking filesystem operations in tests and watching file changes.

## Key Types
- `Filesystem` - Interface abstracting common filesystem operations (Stat, Create, MkdirAll, etc.)
- `File` - Interface for file operations (Write, Sync, Close)
- `DefaultFs` - Production implementation using real filesystem
- `FSWatcher` - Interface for watching filesystem changes
- `fsnotifyWatcher` - Implementation using fsnotify library

## Key Functions
- `NewTempFs()` - Creates a filesystem rooted in a temp directory (for testing)
- `NewFsnotifyWatcher()` - Creates a new filesystem watcher
- `WatchUntil()` - Watches a path for changes with polling fallback
- `MkdirAllWithPathCheck()` - Creates directory with existence check (handles mount points on Windows)

## Design Patterns
- Interface-based filesystem abstraction for testability
- Callback-based watcher pattern with event/error handlers
- Hybrid watching: fsnotify with polling fallback for reliability
- Automatic re-watching on file rename/delete events
- Cross-platform support with platform-specific implementations
