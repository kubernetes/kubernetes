package matchers

import (
	"errors"
	"fmt"

	"github.com/onsi/gomega/format"
)

type MatchErrorStrictlyMatcher struct {
	Expected error
}

func (matcher *MatchErrorStrictlyMatcher) Match(actual any) (success bool, err error) {

	if isNil(matcher.Expected) {
		return false, fmt.Errorf("Expected error is nil, use \"ToNot(HaveOccurred())\" to explicitly check for nil errors")
	}

	if isNil(actual) {
		return false, fmt.Errorf("Expected an error, got nil")
	}

	if !isError(actual) {
		return false, fmt.Errorf("Expected an error.  Got:\n%s", format.Object(actual, 1))
	}

	actualErr := actual.(error)

	return errors.Is(actualErr, matcher.Expected), nil
}

func (matcher *MatchErrorStrictlyMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to match error", matcher.Expected)
}

func (matcher *MatchErrorStrictlyMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to match error", matcher.Expected)
}
