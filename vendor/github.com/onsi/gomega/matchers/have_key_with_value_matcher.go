// untested sections:10

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/matchers/internal/miter"
)

type HaveKeyWithValueMatcher struct {
	Key   any
	Value any
}

func (matcher *HaveKeyWithValueMatcher) Match(actual any) (success bool, err error) {
	if !isMap(actual) && !miter.IsSeq2(actual) {
		return false, fmt.Errorf("HaveKeyWithValue matcher expects a map/iter.Seq2.  Got:%s", format.Object(actual, 1))
	}

	keyMatcher, keyIsMatcher := matcher.Key.(omegaMatcher)
	if !keyIsMatcher {
		keyMatcher = &EqualMatcher{Expected: matcher.Key}
	}

	valueMatcher, valueIsMatcher := matcher.Value.(omegaMatcher)
	if !valueIsMatcher {
		valueMatcher = &EqualMatcher{Expected: matcher.Value}
	}

	if miter.IsSeq2(actual) {
		var success bool
		var err error
		miter.IterateKV(actual, func(k, v reflect.Value) bool {
			success, err = keyMatcher.Match(k.Interface())
			if err != nil {
				err = fmt.Errorf("HaveKey's key matcher failed with:\n%s%s", format.Indent, err.Error())
				return false
			}
			if success {
				success, err = valueMatcher.Match(v.Interface())
				if err != nil {
					err = fmt.Errorf("HaveKeyWithValue's value matcher failed with:\n%s%s", format.Indent, err.Error())
					return false
				}
			}
			return !success
		})
		return success, err
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

func (matcher *HaveKeyWithValueMatcher) FailureMessage(actual any) (message string) {
	str := "to have {key: value}"
	if _, ok := matcher.Key.(omegaMatcher); ok {
		str += " matching"
	} else if _, ok := matcher.Value.(omegaMatcher); ok {
		str += " matching"
	}

	expect := make(map[any]any, 1)
	expect[matcher.Key] = matcher.Value
	return format.Message(actual, str, expect)
}

func (matcher *HaveKeyWithValueMatcher) NegatedFailureMessage(actual any) (message string) {
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
