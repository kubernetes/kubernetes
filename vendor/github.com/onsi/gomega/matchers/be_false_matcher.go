package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type BeFalseMatcher struct {
}

func (matcher *BeFalseMatcher) Match(actual interface{}) (success bool, err error) {
	if !isBool(actual) {
		return false, fmt.Errorf("Expected a boolean.  Got:\n%s", format.Object(actual, 1))
	}

	return actual == false, nil
}

func (matcher *BeFalseMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to be false")
}

func (matcher *BeFalseMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to be false")
}
