package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
	"golang.org/x/xerrors"
)

type MatchErrorMatcher struct {
	Expected interface{}
}

func (matcher *MatchErrorMatcher) Match(actual interface{}) (success bool, err error) {
	if isNil(actual) {
		return false, fmt.Errorf("Expected an error, got nil")
	}

	if !isError(actual) {
		return false, fmt.Errorf("Expected an error.  Got:\n%s", format.Object(actual, 1))
	}

	actualErr := actual.(error)
	expected := matcher.Expected

	if isError(expected) {
		return reflect.DeepEqual(actualErr, expected) || xerrors.Is(actualErr, expected.(error)), nil
	}

	if isString(expected) {
		return actualErr.Error() == expected, nil
	}

	var subMatcher omegaMatcher
	var hasSubMatcher bool
	if expected != nil {
		subMatcher, hasSubMatcher = (expected).(omegaMatcher)
		if hasSubMatcher {
			return subMatcher.Match(actualErr.Error())
		}
	}

	return false, fmt.Errorf(
		"MatchError must be passed an error, a string, or a Matcher that can match on strings. Got:\n%s",
		format.Object(expected, 1))
}

func (matcher *MatchErrorMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to match error", matcher.Expected)
}

func (matcher *MatchErrorMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to match error", matcher.Expected)
}
