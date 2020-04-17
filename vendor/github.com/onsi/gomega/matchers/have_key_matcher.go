// untested sections: 6

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type HaveKeyMatcher struct {
	Key interface{}
}

func (matcher *HaveKeyMatcher) Match(actual interface{}) (success bool, err error) {
	if !isMap(actual) {
		return false, fmt.Errorf("HaveKey matcher expects a map.  Got:%s", format.Object(actual, 1))
	}

	keyMatcher, keyIsMatcher := matcher.Key.(omegaMatcher)
	if !keyIsMatcher {
		keyMatcher = &EqualMatcher{Expected: matcher.Key}
	}

	keys := reflect.ValueOf(actual).MapKeys()
	for i := 0; i < len(keys); i++ {
		success, err := keyMatcher.Match(keys[i].Interface())
		if err != nil {
			return false, fmt.Errorf("HaveKey's key matcher failed with:\n%s%s", format.Indent, err.Error())
		}
		if success {
			return true, nil
		}
	}

	return false, nil
}

func (matcher *HaveKeyMatcher) FailureMessage(actual interface{}) (message string) {
	switch matcher.Key.(type) {
	case omegaMatcher:
		return format.Message(actual, "to have key matching", matcher.Key)
	default:
		return format.Message(actual, "to have key", matcher.Key)
	}
}

func (matcher *HaveKeyMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	switch matcher.Key.(type) {
	case omegaMatcher:
		return format.Message(actual, "not to have key matching", matcher.Key)
	default:
		return format.Message(actual, "not to have key", matcher.Key)
	}
}
