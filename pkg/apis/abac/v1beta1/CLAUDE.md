# Package: v1beta1

## Purpose
Defines the v1beta1 external API version for ABAC policy rules. This is the full-featured version with API group and non-resource path support.

## Key Types

### Policy
```go
type Policy struct {
    metav1.TypeMeta
    Spec PolicySpec
}
```

### PolicySpec
Contains the attributes for a policy rule:
- `User` - Username this rule applies to ("*" matches all)
- `Group` - Group this rule applies to ("*" matches all)
- `Readonly` - When true, matches only readonly requests
- `APIGroup` - API group name ("*" matches all)
- `Resource` - Resource name ("*" matches all)
- `Namespace` - Namespace name ("*" matches all including unnamespaced)
- `NonResourcePath` - Matches non-resource paths ("*" all, "/foo/*" subpaths)

## Differences from v0
- Uses `Spec` wrapper for policy attributes (matches internal version)
- Adds `APIGroup` field for API group matching
- Adds `NonResourcePath` field for non-resource path matching

## Design Notes
- This is the preferred external version for ABAC policies
- Structure mirrors the internal version for straightforward conversion
- All fields are optional and use JSON tags for serialization
