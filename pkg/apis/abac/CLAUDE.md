# Package: abac

## Purpose
Defines the internal types for Attribute-Based Access Control (ABAC) policy rules used by the Kubernetes API server authorization.

## Key Types

### Policy
The main type representing a single ABAC policy rule.
```go
type Policy struct {
    metav1.TypeMeta
    Spec PolicySpec
}
```

### PolicySpec
Contains the attributes that define a policy rule:
- `User` - Username this rule applies to ("*" matches all users)
- `Group` - Group this rule applies to ("*" matches all groups)
- `Readonly` - When true, matches only readonly requests
- `APIGroup` - API group name ("*" matches all)
- `Resource` - Resource name ("*" matches all)
- `Namespace` - Namespace name ("*" matches all including unnamespaced)
- `NonResourcePath` - Matches non-resource paths ("*" all, "/foo/*" subpaths)

## Design Notes
- Either User or Group is required to match a request
- APIGroup, Resource, and Namespace are required together for resource requests
- This is the internal (hub) version; see v0 and v1beta1 for external versions
- ABAC is a legacy authorization mode; RBAC is the recommended alternative
