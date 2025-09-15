package matchers

import (
	"errors"
	"fmt"

	"github.com/onsi/gomega/format"
)

type formattedGomegaError interface {
	FormattedGomegaError() string
}

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
	var fgErr formattedGomegaError
	if errors.As(actual.(error), &fgErr) {
		return fgErr.FormattedGomegaError()
	}
	return fmt.Sprintf("Expected success, but got an error:\n%s", format.Object(actual, 1))
}

func (matcher *SucceedMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return "Expected failure, but got no error."
}
