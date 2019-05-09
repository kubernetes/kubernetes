package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
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

	if isError(matcher.Expected) {
		return reflect.DeepEqual(actualErr, matcher.Expected), nil
	}

	if isString(matcher.Expected) {
		return actualErr.Error() == matcher.Expected, nil
	}

	var subMatcher omegaMatcher
	var hasSubMatcher bool
	if matcher.Expected != nil {
		subMatcher, hasSubMatcher = (matcher.Expected).(omegaMatcher)
		if hasSubMatcher {
			return subMatcher.Match(actualErr.Error())
		}
	}

	return false, fmt.Errorf("MatchError must be passed an error, string, or Matcher that can match on strings.  Got:\n%s", format.Object(matcher.Expected, 1))
}

func (matcher *MatchErrorMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to match error", matcher.Expected)
}

func (matcher *MatchErrorMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to match error", matcher.Expected)
}
