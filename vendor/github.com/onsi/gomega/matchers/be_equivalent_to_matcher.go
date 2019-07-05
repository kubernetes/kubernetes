package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type BeEquivalentToMatcher struct {
	Expected interface{}
}

func (matcher *BeEquivalentToMatcher) Match(actual interface{}) (success bool, err error) {
	if actual == nil && matcher.Expected == nil {
		return false, fmt.Errorf("Both actual and expected must not be nil.")
	}

	convertedActual := actual

	if actual != nil && matcher.Expected != nil && reflect.TypeOf(actual).ConvertibleTo(reflect.TypeOf(matcher.Expected)) {
		convertedActual = reflect.ValueOf(actual).Convert(reflect.TypeOf(matcher.Expected)).Interface()
	}

	return reflect.DeepEqual(convertedActual, matcher.Expected), nil
}

func (matcher *BeEquivalentToMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to be equivalent to", matcher.Expected)
}

func (matcher *BeEquivalentToMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to be equivalent to", matcher.Expected)
}
