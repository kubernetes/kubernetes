package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type BeKeyOfMatcher struct {
	Map any
}

func (matcher *BeKeyOfMatcher) Match(actual any) (success bool, err error) {
	if !isMap(matcher.Map) {
		return false, fmt.Errorf("BeKeyOf matcher needs expected to be a map type")
	}

	if reflect.TypeOf(actual) == nil {
		return false, fmt.Errorf("BeKeyOf matcher expects actual to be typed")
	}

	var lastError error
	for _, key := range reflect.ValueOf(matcher.Map).MapKeys() {
		matcher := &EqualMatcher{Expected: key.Interface()}
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

func (matcher *BeKeyOfMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to be a key of", presentable(valuesOf(matcher.Map)))
}

func (matcher *BeKeyOfMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to be a key of", presentable(valuesOf(matcher.Map)))
}
