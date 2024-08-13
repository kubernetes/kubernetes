// untested sections: 2

package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type HaveOccurredMatcher struct {
}

func (matcher *HaveOccurredMatcher) Match(actual interface{}) (success bool, err error) {
	// is purely nil?
	if actual == nil {
		return false, nil
	}

	// must be an 'error' type
	if !isError(actual) {
		return false, fmt.Errorf("Expected an error-type.  Got:\n%s", format.Object(actual, 1))
	}

	// must be non-nil (or a pointer to a non-nil)
	return !isNil(actual), nil
}

func (matcher *HaveOccurredMatcher) FailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("Expected an error to have occurred.  Got:\n%s", format.Object(actual, 1))
}

func (matcher *HaveOccurredMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("Unexpected error:\n%s\n%s", format.Object(actual, 1), "occurred")
}
