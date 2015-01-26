package matchers

import (
	"fmt"
	"github.com/onsi/gomega/format"
)

type HaveOccurredMatcher struct {
}

func (matcher *HaveOccurredMatcher) Match(actual interface{}) (success bool, err error) {
	if actual == nil {
		return false, nil
	}

	if isError(actual) {
		return true, nil
	}

	return false, fmt.Errorf("Expected an error.  Got:\n%s", format.Object(actual, 1))
}

func (matcher *HaveOccurredMatcher) FailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("Expected an error to have occured.  Got:\n%s", format.Object(actual, 1))
}

func (matcher *HaveOccurredMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("Expected error:\n%s\n%s\n%s", format.Object(actual, 1), format.IndentString(actual.(error).Error(), 1), "not to have occurred")
}
