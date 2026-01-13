# Package: taints

## Purpose
Provides utilities for working with Kubernetes node taints, including parsing, validation, and manipulation.

## Key Functions
- `ParseTaints()` - Parses taint specifications into add/remove lists
- `AddOrUpdateTaint()` - Adds or updates a taint on a node (returns new copy)
- `RemoveTaint()` - Removes a taint from a node (returns new copy)
- `DeleteTaint()` / `DeleteTaintsByKey()` - Removes taints from a list
- `TaintExists()` / `TaintKeyExists()` - Checks for taint presence
- `TaintSetDiff()` - Finds added/removed taints between two lists
- `TaintSetFilter()` - Filters taints by predicate function
- `CheckTaintValidation()` - Validates taint key, value, and effect

## Taint String Formats
- `key=value:effect` - Full taint specification
- `key:effect` - Taint without value
- `key` - Taint key only (for removal)
- `key-` suffix indicates removal

## Valid Effects
- NoSchedule, PreferNoSchedule, NoExecute

## Design Patterns
- Non-mutating: functions return new nodes/slices
- Validates against Kubernetes naming conventions
- Supports bulk operations via taint sets
- Used by kubectl taint and node controllers
