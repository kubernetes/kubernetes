package types

// Versions and Prefixes used in API and KV URLs
const (

	// RuleActionAdd specifies to add labels to the object(s).
	RuleActionAdd RuleAction = "add"

	// RuleActionRemove specifies to remove labels from the object(s).
	RuleActionRemove RuleAction = "remove"
)

// RuleAction - rule action type
type RuleAction string

// Rule is used to define a rule
type Rule struct {

	// Rule unique ID.
	// Read Only: true
	ID string `json:"id"`

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
}

// Rules is a collection of Rules.
type Rules []*Rule
