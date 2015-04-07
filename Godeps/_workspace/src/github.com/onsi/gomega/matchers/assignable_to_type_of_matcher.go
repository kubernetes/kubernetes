package matchers

import (
	"fmt"
	"github.com/onsi/gomega/format"
	"reflect"
)

type AssignableToTypeOfMatcher struct {
	Expected interface{}
}

func (matcher *AssignableToTypeOfMatcher) Match(actual interface{}) (success bool, err error) {
	if actual == nil || matcher.Expected == nil {
		return false, fmt.Errorf("Refusing to compare <nil> to <nil>.")
	}

	actualType := reflect.TypeOf(actual)
	expectedType := reflect.TypeOf(matcher.Expected)

	return actualType.AssignableTo(expectedType), nil
}

func (matcher *AssignableToTypeOfMatcher) FailureMessage(actual interface{}) string {
	return format.Message(actual, fmt.Sprintf("to be assignable to the type: %T", matcher.Expected))
}

func (matcher *AssignableToTypeOfMatcher) NegatedFailureMessage(actual interface{}) string {
	return format.Message(actual, fmt.Sprintf("not to be assignable to the type: %T", matcher.Expected))
}
