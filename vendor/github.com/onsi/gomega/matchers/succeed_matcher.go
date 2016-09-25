package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
)

type SucceedMatcher struct {
}

func (matcher *SucceedMatcher) Match(actual interface{}) (success bool, err error) {
	// is purely nil?
	if actual == nil {
		return true, nil
	}

	// must be an 'error' type
	if !isError(actual) {
		return false, fmt.Errorf("Expected an error-type.  Got:\n%s", format.Object(actual, 1))
	}

	// must be nil (or a pointer to a nil)
	return isNil(actual), nil
}

func (matcher *SucceedMatcher) FailureMessage(actual interface{}) (message string) {
	return fmt.Sprintf("Expected success, but got an error:\n%s\n%s", format.Object(actual, 1), format.IndentString(actual.(error).Error(), 1))
}

func (matcher *SucceedMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return "Expected failure, but got no error."
}
