# Package: hash

## Purpose
Provides utilities for creating deterministic hashes of Go objects, useful for detecting changes in complex structures.

## Key Functions
- `DeepHashObject()` - Writes an object to a hash, following pointers to actual values

## Design Patterns
- Uses k8s.io/apimachinery/pkg/util/dump for consistent object serialization
- Follows pointers to ensure hash reflects actual data, not pointer addresses
- Hash remains stable when only pointers change but values don't
- Resets hasher before writing for clean state
- Commonly used for config change detection and cache keys
