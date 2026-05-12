package matchers

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type MatchErrorMatcher struct {
	Expected           any
	FuncErrDescription []any
	isFunc             bool
}

func (matcher *MatchErrorMatcher) Match(actual any) (success bool, err error) {
	matcher.isFunc = false

	if isNil(actual) {
		return false, fmt.Errorf("Expected an error, got nil")
	}

	if !isError(actual) {
		return false, fmt.Errorf("Expected an error.  Got:\n%s", format.Object(actual, 1))
	}

	actualErr := actual.(error)
	expected := matcher.Expected

	if isError(expected) {
		// first try the built-in errors.Is
		if errors.Is(actualErr, expected.(error)) {
			return true, nil
		}
		// if not, try DeepEqual along the error chain
		for unwrapped := actualErr; unwrapped != nil; unwrapped = errors.Unwrap(unwrapped) {
			if reflect.DeepEqual(unwrapped, expected) {
				return true, nil
			}
		}
		return false, nil
	}

	if isString(expected) {
		return actualErr.Error() == expected, nil
	}

	v := reflect.ValueOf(expected)
	t := v.Type()
	errorInterface := reflect.TypeOf((*error)(nil)).Elem()
	if t.Kind() == reflect.Func && t.NumIn() == 1 && t.In(0).Implements(errorInterface) && t.NumOut() == 1 && t.Out(0).Kind() == reflect.Bool {
		if len(matcher.FuncErrDescription) == 0 {
			return false, fmt.Errorf("MatchError requires an additional description when passed a function")
		}
		matcher.isFunc = true
		return v.Call([]reflect.Value{reflect.ValueOf(actualErr)})[0].Bool(), nil
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

func (matcher *MatchErrorMatcher) FailureMessage(actual any) (message string) {
	if matcher.isFunc {
		return format.Message(actual, fmt.Sprintf("to match error function %s", matcher.FuncErrDescription[0]))
	}
	return format.Message(actual, "to match error", matcher.Expected)
}

func (matcher *MatchErrorMatcher) NegatedFailureMessage(actual any) (message string) {
	if matcher.isFunc {
		return format.Message(actual, fmt.Sprintf("not to match error function %s", matcher.FuncErrDescription[0]))
	}
	return format.Message(actual, "not to match error", matcher.Expected)
}
