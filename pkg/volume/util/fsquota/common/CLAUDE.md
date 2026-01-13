# Package: common

## Purpose
Defines common types and constants shared across fsquota implementations.

## Key Types/Structs
- `QuotaID` - int32 type representing a filesystem quota identifier

## Key Constants
- `UnknownQuotaID` (-1) - Cannot determine if quota is in force
- `BadQuotaID` (0) - Invalid quota identifier

## Design Patterns
- Shared types to avoid circular dependencies
- Based on quotactl(2) data types
- Used by both Linux implementation and interface definitions
