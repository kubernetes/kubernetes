# Package: validation

## Purpose
Provides validation logic for authorization API types including SubjectAccessReview, SelfSubjectAccessReview, and LocalSubjectAccessReview.

## Key Functions
- `ValidateSubjectAccessReview(sar *SubjectAccessReview)`: Validates a SubjectAccessReview request
- `ValidateSelfSubjectAccessReview(sar *SelfSubjectAccessReview)`: Validates a SelfSubjectAccessReview request
- `ValidateLocalSubjectAccessReview(sar *LocalSubjectAccessReview)`: Validates a LocalSubjectAccessReview (namespace-scoped)
- `ValidateSubjectAccessReviewSpec(spec, fldPath)`: Validates the spec portion of an access review
- `ValidateSelfSubjectAccessReviewSpec(spec, fldPath)`: Validates self-subject review spec
- `validateResourceAttributes(resourceAttributes, fldPath)`: Validates resource access attributes
- `validateFieldSelectorAttributes(selector, fldPath)`: Validates field selector attributes
- `validateLabelSelectorAttributes(selector, fldPath)`: Validates label selector attributes

## Validation Rules
- Exactly one of ResourceAttributes or NonResourceAttributes must be specified
- For SubjectAccessReview, at least one of user or groups must be specified
- Metadata must be empty (except namespace for LocalSubjectAccessReview)
- LocalSubjectAccessReview namespace must match spec.resourceAttributes.namespace
- LocalSubjectAccessReview cannot have NonResourceAttributes
- Field/label selectors cannot have both rawSelector and requirements set
- When selector is specified, requirements or rawSelector is required
