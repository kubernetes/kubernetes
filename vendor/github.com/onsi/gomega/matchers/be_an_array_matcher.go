// untested sections: 1

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type BeAnArrayMatcher struct {
}

func (matcher *BeAnArrayMatcher) Match(actual any) (success bool, err error) {
	if actual == nil {
		return false, fmt.Errorf("BeAnArray matcher expects a value, got nil")
	}
	return reflect.TypeOf(actual).Kind() == reflect.Array, nil
}

func (matcher *BeAnArrayMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to be an array")
}

func (matcher *BeAnArrayMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to be an array")
}
