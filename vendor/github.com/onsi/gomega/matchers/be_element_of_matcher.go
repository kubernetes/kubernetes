// untested sections: 1

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type BeElementOfMatcher struct {
	Elements []any
}

func (matcher *BeElementOfMatcher) Match(actual any) (success bool, err error) {
	if reflect.TypeOf(actual) == nil {
		return false, fmt.Errorf("BeElement matcher expects actual to be typed")
	}

	var lastError error
	for _, m := range flatten(matcher.Elements) {
		matcher := &EqualMatcher{Expected: m}
		success, err := matcher.Match(actual)
		if err != nil {
			lastError = err
			continue
		}
		if success {
			return true, nil
		}
	}

	return false, lastError
}

func (matcher *BeElementOfMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to be an element of", presentable(matcher.Elements))
}

func (matcher *BeElementOfMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to be an element of", presentable(matcher.Elements))
}
