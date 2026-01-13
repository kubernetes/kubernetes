# Package: validation

## Purpose
Provides validation utilities for PersistentVolume objects, specifically plugin-specific validation beyond core API validation.

## Key Types/Structs
- None (utility package)

## Key Functions
- `ValidatePersistentVolume()` - Plugin-specific PV validation
- `checkMountOption()` - Validates mount options are supported by volume type
- `ValidatePathNoBacksteps()` - Ensures paths don't contain ".." elements

## Design Patterns
- Complements core API validation with plugin-specific rules
- Validates mount options are only used with supporting volume types
- Security validation to prevent path traversal attacks
- Returns field.ErrorList for consistent error reporting
