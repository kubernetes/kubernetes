// untested sections: 2

package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type BeFalseMatcher struct {
	Reason string
}

func (matcher *BeFalseMatcher) Match(actual any) (success bool, err error) {
	if !isBool(actual) {
		return false, fmt.Errorf("Expected a boolean.  Got:\n%s", format.Object(actual, 1))
	}

	return actual == false, nil
}

func (matcher *BeFalseMatcher) FailureMessage(actual any) (message string) {
	if matcher.Reason == "" {
		return format.Message(actual, "to be false")
	} else {
		return matcher.Reason
	}
}

func (matcher *BeFalseMatcher) NegatedFailureMessage(actual any) (message string) {
	if matcher.Reason == "" {
		return format.Message(actual, "not to be false")
	} else {
		return fmt.Sprintf(`Expected not false but got false\nNegation of "%s" failed`, matcher.Reason)
	}
}
