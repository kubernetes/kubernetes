// untested sections: 6

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/matchers/internal/miter"
)

type HaveKeyMatcher struct {
	Key any
}

func (matcher *HaveKeyMatcher) Match(actual any) (success bool, err error) {
	if !isMap(actual) && !miter.IsSeq2(actual) {
		return false, fmt.Errorf("HaveKey matcher expects a map/iter.Seq2.  Got:%s", format.Object(actual, 1))
	}

	keyMatcher, keyIsMatcher := matcher.Key.(omegaMatcher)
	if !keyIsMatcher {
		keyMatcher = &EqualMatcher{Expected: matcher.Key}
	}

	// As with ContainElement, a key the key matcher accepts wins regardless of
	// errors on other keys; the matcher's (last) error is only reported when
	// no key matches.
	var lastError error
	if miter.IsSeq2(actual) {
		found := false
		miter.IterateKV(actual, func(k, v reflect.Value) bool {
			success, err := keyMatcher.Match(k.Interface())
			if err != nil {
				lastError = err
				return true
			}
			found = success
			return !found
		})
		if found {
			return true, nil
		}
	} else {
		keys := reflect.ValueOf(actual).MapKeys()
		for i := range keys {
			success, err := keyMatcher.Match(keys[i].Interface())
			if err != nil {
				lastError = err
				continue
			}
			if success {
				return true, nil
			}
		}
	}

	if lastError != nil {
		return false, fmt.Errorf("HaveKey's key matcher failed with:\n%s%s", format.Indent, lastError.Error())
	}
	return false, nil
}

func (matcher *HaveKeyMatcher) FailureMessage(actual any) (message string) {
	switch matcher.Key.(type) {
	case omegaMatcher:
		return format.Message(actual, "to have key matching", matcher.Key)
	default:
		return format.Message(actual, "to have key", matcher.Key)
	}
}

func (matcher *HaveKeyMatcher) NegatedFailureMessage(actual any) (message string) {
	switch matcher.Key.(type) {
	case omegaMatcher:
		return format.Message(actual, "not to have key matching", matcher.Key)
	default:
		return format.Message(actual, "not to have key", matcher.Key)
	}
}
