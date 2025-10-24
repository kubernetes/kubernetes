# Taint Management

## Brief Overview
Parse, validate, and manipulate Kubernetes node taints with support for three string formats and immutable node operations.

## Available Dependencies
- k8s.io/api: v0.0.0
- k8s.io/apimachinery: v0.0.0
- k8s.io/kubernetes/pkg/apis/core/helper: (internal)

## Core Requirements

### Taint String Parsing

Parse taint strings in three formats:
- `key=value:effect` - full taint with key, value, and effect
- `key:effect` - taint with key and effect only (empty value)
- `key` - key only (empty value and effect)

Validate components:
- Key: qualified name format (validation via `k8s.io/apimachinery/pkg/util/validation.IsQualifiedName`)
- Value: label value format when present (validation via `k8s.io/apimachinery/pkg/util/validation.IsValidLabelValue`)
- Effect: exactly one of `NoSchedule`, `PreferNoSchedule`, or `NoExecute`

Reject malformed input:
- More than one `:` separator
- More than one `=` separator in key-value portion
- Invalid characters or length in key or value

### Batch Taint Parsing

Parse array of taint specifications supporting add and remove operations:
- Remove syntax: append `-` suffix to any format (e.g., `key:effect-`, `key-`)
- Add operations require an effect; reject format `key` without effect
- Reject duplicate taints with same key and effect within add operations
- Return separate lists: taints to add and taints to remove (removal list includes only key and effect)

### Node Taint Modification

Immutable operations returning new node copies:
- Add: append taint if key-effect combination doesn't exist
- Update: replace existing taint with same key and effect if value differs
- No-op: return unchanged node if taint already exists with identical value
- Remove: delete taint matching key and effect if present

Return boolean indicating whether modification occurred.

### Taint Collection Operations

Check existence by matching key and effect (value irrelevant for matching).

Delete taints:
- By key: remove all taints with matching key regardless of effect
- By key and effect: remove taints matching both

Compute set difference between two taint lists identifying elements in first but not second (to add) and elements in second but not first (to remove).

Filter taints using predicate function retaining elements where function returns true.

Check for pre-existing taints by key-effect pairs returning comma-separated string of matching keys.

### Taint Validation

Validate individual taint components independently:
- Empty key is invalid
- Value exceeding 63 characters is invalid
- Effect must be valid enum value when non-empty
- Empty value and empty effect are valid
