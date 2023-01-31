package v4

import (
	"github.com/aws/aws-sdk-go/internal/strings"
)

// validator houses a set of rule needed for validation of a
// string value
type rules []rule

// rule interface allows for more flexible rules and just simply
// checks whether or not a value adheres to that rule
type rule interface {
	IsValid(value string) bool
}

// IsValid will iterate through all rules and see if any rules
// apply to the value and supports nested rules
func (r rules) IsValid(value string) bool {
	for _, rule := range r {
		if rule.IsValid(value) {
			return true
		}
	}
	return false
}

// mapRule generic rule for maps
type mapRule map[string]struct{}

// IsValid for the map rule satisfies whether it exists in the map
func (m mapRule) IsValid(value string) bool {
	_, ok := m[value]
	return ok
}

// allowList is a generic rule for allow listing
type allowList struct {
	rule
}

// IsValid for allow list checks if the value is within the allow list
func (w allowList) IsValid(value string) bool {
	return w.rule.IsValid(value)
}

// excludeList is a generic rule for exclude listing
type excludeList struct {
	rule
}

// IsValid for exclude list checks if the value is within the exclude list
func (b excludeList) IsValid(value string) bool {
	return !b.rule.IsValid(value)
}

type patterns []string

// IsValid for patterns checks each pattern and returns if a match has
// been found
func (p patterns) IsValid(value string) bool {
	for _, pattern := range p {
		if strings.HasPrefixFold(value, pattern) {
			return true
		}
	}
	return false
}

// inclusiveRules rules allow for rules to depend on one another
type inclusiveRules []rule

// IsValid will return true if all rules are true
func (r inclusiveRules) IsValid(value string) bool {
	for _, rule := range r {
		if !rule.IsValid(value) {
			return false
		}
	}
	return true
}
