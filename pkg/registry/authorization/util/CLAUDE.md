# Package: util

## Purpose
Provides utility functions for converting authorization API objects to authorizer.AttributesRecord for use in authorization decisions.

## Key Functions

- **ResourceAttributesFrom(user, in)**: Converts ResourceAttributes + user to authorizer.AttributesRecord
  - Handles label and field selectors when AuthorizeWithSelectors feature gate is enabled
  - Sets ResourceRequest=true
- **NonResourceAttributesFrom(user, in)**: Converts NonResourceAttributes + user to authorizer.AttributesRecord
  - Sets ResourceRequest=false, includes Path and Verb
- **AuthorizationAttributesFrom(spec)**: Converts SubjectAccessReviewSpec to authorizer.AttributesRecord
  - Creates user.DefaultInfo from spec (user, groups, uid, extra)
  - Delegates to ResourceAttributesFrom or NonResourceAttributesFrom based on spec
- **BuildEvaluationError(evaluationError, attrs)**: Constructs evaluation error string
  - Includes authorizer error and any selector parsing errors

## Helper Functions

- **labelSelectorAsSelector()**: Converts LabelSelectorRequirements to labels.Requirements
- **fieldSelectorAsSelector()**: Converts FieldSelectorRequirements to fields.Requirements
- **matchAllVersionIfEmpty()**: Returns "*" if version is empty (match all versions)
- **convertToUserInfoExtra()**: Converts ExtraValue map to string slice map

## Design Notes

- Supports AuthorizeWithSelectors feature gate for granular authorization
- Selector parsing errors are aggregated but valid requirements still returned
- Empty API version matches all versions ("*")
