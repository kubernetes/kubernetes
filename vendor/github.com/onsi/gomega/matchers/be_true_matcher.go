package matchers

import (
	"fmt"
	"github.com/onsi/gomega/format"
)

type BeTrueMatcher struct {
}

func (matcher *BeTrueMatcher) Match(actual interface{}) (success bool, err error) {
	if !isBool(actual) {
		return false, fmt.Errorf("Expected a boolean.  Got:\n%s", format.Object(actual, 1))
	}

	return actual.(bool), nil
}

func (matcher *BeTrueMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to be true")
}

func (matcher *BeTrueMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to be true")
}
