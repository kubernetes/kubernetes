package types

import "context"

// RuleCreateOptions are available parameters for creating new rules.
type RuleCreateOptions struct {

	// Rule name.
	// Required: true
	Name string `json:"name"`

	// Rule description.
	Description string `json:"description"`

	// Flag describing whether the rule is active.
	// Default: false
	Active bool `json:"active"`

	// Weight is used to determine order during rule processing.  Rules with
	// heavier weights are processed later.
	// default: 0
	Weight int `json:"weight"`

	// Operator is used to compare objects or labels.
	Operator Operator `json:"operator"`

	// RuleAction controls whether the action is to add or remove a label from the
	// matching object(s).
	RuleAction RuleAction `json:"ruleAction"`

	// Selectors defines the list of labels that should trigger a rule.
	Selectors map[string]string `json:"selectors"`

	// Labels define the list of labels that will be added or removed from the
	// matching object(s).
	Labels map[string]string `json:"labels"`

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context `json:"-"`
}
