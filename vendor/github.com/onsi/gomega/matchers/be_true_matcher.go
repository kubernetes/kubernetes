// untested sections: 2

package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type BeTrueMatcher struct {
	Reason string
}

func (matcher *BeTrueMatcher) Match(actual any) (success bool, err error) {
	if !isBool(actual) {
		return false, fmt.Errorf("Expected a boolean.  Got:\n%s", format.Object(actual, 1))
	}

	return actual.(bool), nil
}

func (matcher *BeTrueMatcher) FailureMessage(actual any) (message string) {
	if matcher.Reason == "" {
		return format.Message(actual, "to be true")
	} else {
		return matcher.Reason
	}
}

func (matcher *BeTrueMatcher) NegatedFailureMessage(actual any) (message string) {
	if matcher.Reason == "" {
		return format.Message(actual, "not to be true")
	} else {
		return fmt.Sprintf(`Expected not true but got true\nNegation of "%s" failed`, matcher.Reason)
	}
}
