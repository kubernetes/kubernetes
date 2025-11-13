package matchers

import (
	"reflect"

	"github.com/onsi/gomega/format"
)

type BeZeroMatcher struct {
}

func (matcher *BeZeroMatcher) Match(actual any) (success bool, err error) {
	if actual == nil {
		return true, nil
	}
	zeroValue := reflect.Zero(reflect.TypeOf(actual)).Interface()

	return reflect.DeepEqual(zeroValue, actual), nil

}

func (matcher *BeZeroMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to be zero-valued")
}

func (matcher *BeZeroMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to be zero-valued")
}
