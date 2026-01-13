# Package: env

## Purpose
Provides helper functions for reading environment variables with fallback default values.

## Key Functions
- `GetEnvAsStringOrFallback()` - Returns env var value or default string
- `GetEnvAsIntOrFallback()` - Returns env var as int or default, with error handling
- `GetEnvAsFloat64OrFallback()` - Returns env var as float64 or default, with error handling

## Design Patterns
- Simple wrapper functions around os.Getenv
- Type conversion with error propagation
- Empty string treated as unset (returns default)
- Commonly used for configuration from environment
