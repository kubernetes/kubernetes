package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type HaveEachMatcher struct {
	Element interface{}
}

func (matcher *HaveEachMatcher) Match(actual interface{}) (success bool, err error) {
	if !isArrayOrSlice(actual) && !isMap(actual) {
		return false, fmt.Errorf("HaveEach matcher expects an array/slice/map.  Got:\n%s",
			format.Object(actual, 1))
	}

	elemMatcher, elementIsMatcher := matcher.Element.(omegaMatcher)
	if !elementIsMatcher {
		elemMatcher = &EqualMatcher{Expected: matcher.Element}
	}

	value := reflect.ValueOf(actual)
	if value.Len() == 0 {
		return false, fmt.Errorf("HaveEach matcher expects a non-empty array/slice/map.  Got:\n%s",
			format.Object(actual, 1))
	}

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

	// if there are no elements, then HaveEach will match.
	for i := 0; i < value.Len(); i++ {
		success, err := elemMatcher.Match(valueAt(i))
		if err != nil {
			return false, err
		}
		if !success {
			return false, nil
		}
	}

	return true, nil
}

// FailureMessage returns a suitable failure message.
func (matcher *HaveEachMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to contain element matching", matcher.Element)
}

// NegatedFailureMessage returns a suitable negated failure message.
func (matcher *HaveEachMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to contain element matching", matcher.Element)
}
