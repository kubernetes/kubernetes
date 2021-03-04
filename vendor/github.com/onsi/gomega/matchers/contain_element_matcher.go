// untested sections: 2

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type ContainElementMatcher struct {
	Element interface{}
}

func (matcher *ContainElementMatcher) Match(actual interface{}) (success bool, err error) {
	if !isArrayOrSlice(actual) && !isMap(actual) {
		return false, fmt.Errorf("ContainElement matcher expects an array/slice/map.  Got:\n%s", format.Object(actual, 1))
	}

	elemMatcher, elementIsMatcher := matcher.Element.(omegaMatcher)
	if !elementIsMatcher {
		elemMatcher = &EqualMatcher{Expected: matcher.Element}
	}

	value := reflect.ValueOf(actual)
	var valueAt func(int) interface{}
	if isMap(actual) {
		keys := value.MapKeys()
		valueAt = func(i int) interface{} {
			return value.MapIndex(keys[i]).Interface()
		}
	} else {
		valueAt = func(i int) interface{} {
			return value.Index(i).Interface()
		}
	}

	var lastError error
	for i := 0; i < value.Len(); i++ {
		success, err := elemMatcher.Match(valueAt(i))
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

func (matcher *ContainElementMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to contain element matching", matcher.Element)
}

func (matcher *ContainElementMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to contain element matching", matcher.Element)
}
