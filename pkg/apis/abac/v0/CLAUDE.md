# Package: v0

## Purpose
Defines the v0 (legacy) external API version for ABAC policy rules. This is a simplified format without API group support.

## Key Types

### Policy
```go
type Policy struct {
    metav1.TypeMeta
    User      string  // Username ("*" matches all)
    Group     string  // Group ("*" matches all)
    Readonly  bool    // Match readonly requests only
    Resource  string  // Resource name ("*" matches all)
    Namespace string  // Namespace ("*" matches all)
}
```

## Differences from v1beta1
- No `Spec` wrapper - fields are at the top level
- No `APIGroup` field - cannot match specific API groups
- No `NonResourcePath` field - cannot match non-resource paths

## Design Notes
- This is the original ABAC policy format
- Conversion functions exist to convert to/from the internal version
- v1beta1 is the preferred external version with full feature support
