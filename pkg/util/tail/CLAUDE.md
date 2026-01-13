# Package: tail

## Purpose
Provides utilities for reading the end of files, similar to the Unix `tail` command.

## Key Functions
- `ReadAtMost()` - Reads at most N bytes from the end of a file

## Return Values
- `[]byte` - The data read (up to max bytes)
- `bool` - True if file was longer than max (data was truncated)
- `error` - Any error encountered

## Implementation Details
- Opens file and seeks to end minus max bytes
- Reads from seek position to end
- Returns empty slice for empty files
- Uses 1KB block size internally

## Design Patterns
- Efficient for large files (only reads needed portion)
- Returns truncation indicator for caller awareness
- Used by log readers to get recent log entries
