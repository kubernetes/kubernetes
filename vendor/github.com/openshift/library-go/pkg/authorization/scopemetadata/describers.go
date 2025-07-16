package scopemetadata

// ScopeDescriber takes a scope and returns metadata about it
type ScopeDescriber interface {
	// Handles returns true if this evaluator can evaluate this scope
	Handles(scope string) bool
	// Validate returns an error if the scope is malformed
	Validate(scope string) error
	// Describe returns a description, warning (typically used to warn about escalation dangers), or an error if the scope is malformed
	Describe(scope string) (description string, warning string, err error)
}

// ScopeDescribers map prefixes to a function that handles that prefix
var ScopeDescribers = []ScopeDescriber{
	UserEvaluator{},
	ClusterRoleEvaluator{},
}
