# Package: exclusion

## Purpose
This package defines lists of resources that expression-based admission controllers (like ValidatingAdmissionPolicy) should include or exclude from interception. It prevents admission policies from breaking cluster functionality.

## Key Functions

- **Included()**: Returns resources that should be intercepted by expression-based admission
- **Excluded()**: Returns resources that should never be intercepted

## Included Resources

Non-persisted resources that are safe to intercept:
- bindings, pods/attach, pods/binding, pods/eviction
- pods/exec, pods/portforward
- serviceaccounts/token

## Excluded Resources

Non-persisted resources that must not be intercepted (would break cluster):
- authentication.k8s.io: selfsubjectreviews, tokenreviews
- authorization.k8s.io: localsubjectaccessreviews, selfsubjectaccessreviews, selfsubjectrulesreviews, subjectaccessreviews

## Design Notes

- Version is omitted; all versions of the same GroupResource are treated identically
- Transient (non-persisted) resources must be in either the included or excluded list
- Returns cloned slices to prevent modification of internal lists
- Excluding auth-related resources prevents infinite loops in authorization checks
