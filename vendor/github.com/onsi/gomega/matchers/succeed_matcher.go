package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type SucceedMatcher struct {
}

func (matcher *SucceedMatcher) Match(actual interface{}) (success bool, err error) {
	if actual == nil {
		return true, nil
	}

	if isError(actual) {
		return false, nil
	}

	return false, fmt.Errorf("Expected an error-type.  Got:\n%s", format.Object(actual, 1))
}

func (matcher *SucceedMatcher) FailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("Expected success, but got an error:\n%s", format.Object(actual, 1))
}

func (matcher *SucceedMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return "Expected failure, but got no error."
}
