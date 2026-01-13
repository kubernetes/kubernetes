# Package: selfsubjectrulesreview

## Purpose
Implements the REST endpoint for SelfSubjectRulesReview, allowing users to list all permissions they have in a specific namespace.

## Key Types

- **REST**: Implements the SelfSubjectRulesReview REST endpoint
  - ruleResolver: The authorizer.RuleResolver to enumerate permissions

## Key Functions

- **NewREST(ruleResolver)**: Creates REST handler with the rule resolver
- **NamespaceScoped()**: Returns false - cluster-scoped resource
- **Create()**: Returns all resource and non-resource rules for the user in the specified namespace
- **GetSingularName()**: Returns "selfsubjectrulesreview"
- **getResourceRules()**: Converts ResourceRuleInfo to API ResourceRule
- **getNonResourceRules()**: Converts NonResourceRuleInfo to API NonResourceRule

## Design Notes

- Create-only resource - no persistent storage
- Requires namespace in spec (unlike SelfSubjectAccessReview)
- Returns lists of ResourceRules (verbs, apiGroups, resources, resourceNames) and NonResourceRules (verbs, URLs)
- Status includes "incomplete" flag if the rule list may be partial
- Uses ruleResolver.RulesFor() to enumerate permissions
- Useful for "kubectl auth can-i --list" functionality
