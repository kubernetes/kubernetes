// untested sections: 1

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type BeASliceMatcher struct {
}

func (matcher *BeASliceMatcher) Match(actual any) (success bool, err error) {
	if actual == nil {
		return false, fmt.Errorf("BeASlice matcher expects a value, got nil")
	}
	return reflect.TypeOf(actual).Kind() == reflect.Slice, nil
}

func (matcher *BeASliceMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to be a slice")
}

func (matcher *BeASliceMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to be a slice")
}
