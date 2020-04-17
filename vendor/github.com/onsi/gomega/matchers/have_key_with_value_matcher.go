// untested sections:10

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type HaveKeyWithValueMatcher struct {
	Key   interface{}
	Value interface{}
}

func (matcher *HaveKeyWithValueMatcher) Match(actual interface{}) (success bool, err error) {
	if !isMap(actual) {
		return false, fmt.Errorf("HaveKeyWithValue matcher expects a map.  Got:%s", format.Object(actual, 1))
	}

	keyMatcher, keyIsMatcher := matcher.Key.(omegaMatcher)
	if !keyIsMatcher {
		keyMatcher = &EqualMatcher{Expected: matcher.Key}
	}

	valueMatcher, valueIsMatcher := matcher.Value.(omegaMatcher)
	if !valueIsMatcher {
		valueMatcher = &EqualMatcher{Expected: matcher.Value}
	}

	keys := reflect.ValueOf(actual).MapKeys()
	for i := 0; i < len(keys); i++ {
		success, err := keyMatcher.Match(keys[i].Interface())
		if err != nil {
			return false, fmt.Errorf("HaveKeyWithValue's key matcher failed with:\n%s%s", format.Indent, err.Error())
		}
		if success {
			actualValue := reflect.ValueOf(actual).MapIndex(keys[i])
			success, err := valueMatcher.Match(actualValue.Interface())
			if err != nil {
				return false, fmt.Errorf("HaveKeyWithValue's value matcher failed with:\n%s%s", format.Indent, err.Error())
			}
			return success, nil
		}
	}

	return false, nil
}

func (matcher *HaveKeyWithValueMatcher) FailureMessage(actual interface{}) (message string) {
	str := "to have {key: value}"
	if _, ok := matcher.Key.(omegaMatcher); ok {
		str += " matching"
	} else if _, ok := matcher.Value.(omegaMatcher); ok {
		str += " matching"
	}

	expect := make(map[interface{}]interface{}, 1)
	expect[matcher.Key] = matcher.Value
	return format.Message(actual, str, expect)
}

func (matcher *HaveKeyWithValueMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	kStr := "not to have key"
	if _, ok := matcher.Key.(omegaMatcher); ok {
		kStr = "not to have key matching"
	}

	vStr := "or that key's value not be"
	if _, ok := matcher.Value.(omegaMatcher); ok {
		vStr = "or to have that key's value not matching"
	}

	return format.Message(actual, kStr, matcher.Key, vStr, matcher.Value)
}
