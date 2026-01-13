# Package: abac

## Purpose
Implements Attribute-Based Access Control (ABAC) authorization for Kubernetes API requests by loading and evaluating policy rules from a file.

## Key Types/Structs
- `PolicyList`: A slice of Policy structs that implements the authorizer.Authorizer interface
- `policyLoadError`: Error type for policy file parsing errors with line number information

## Key Functions
- `NewFromFile(path string)`: Loads policies from a file, supporting both versioned and unversioned policy formats
- `Authorize(ctx, attributes)`: Evaluates all policies and returns DecisionAllow if any match, otherwise DecisionNoOpinion
- `RulesFor(ctx, user, namespace)`: Returns resource and non-resource rules applicable to a user/namespace
- `matches(policy, attributes)`: Checks if a policy matches given authorization attributes
- `subjectMatches`: Matches user/group against policy subject
- `verbMatches`: Matches read-only or all verbs
- `resourceMatches`: Matches namespace, resource, and API group
- `nonResourceMatches`: Matches non-resource paths with wildcard support

## Design Notes
- File format is one policy per line (JSON), supporting comments and blank lines
- Supports wildcard (*) matching for users, groups, namespaces, resources, and paths
- Automatically migrates unversioned (v0) policies to current format
- Read-only policies match only get/list/watch verbs
