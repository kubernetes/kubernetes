package matchers

import (
	"bytes"
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type EqualMatcher struct {
	Expected interface{}
}

func (matcher *EqualMatcher) Match(actual interface{}) (success bool, err error) {
	if actual == nil && matcher.Expected == nil {
		return false, fmt.Errorf("Refusing to compare <nil> to <nil>.\nBe explicit and use BeNil() instead.  This is to avoid mistakes where both sides of an assertion are erroneously uninitialized.")
	}
	// Shortcut for byte slices.
	// Comparing long byte slices with reflect.DeepEqual is very slow,
	// so use bytes.Equal if actual and expected are both byte slices.
	if actualByteSlice, ok := actual.([]byte); ok {
		if expectedByteSlice, ok := matcher.Expected.([]byte); ok {
			return bytes.Equal(actualByteSlice, expectedByteSlice), nil
		}
	}
	return reflect.DeepEqual(actual, matcher.Expected), nil
}

func (matcher *EqualMatcher) FailureMessage(actual interface{}) (message string) {
	actualString, actualOK := actual.(string)
	expectedString, expectedOK := matcher.Expected.(string)
	if actualOK && expectedOK {
		return format.MessageWithDiff(actualString, "to equal", expectedString)
	}

	return format.Message(actual, "to equal", matcher.Expected)
}

func (matcher *EqualMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to equal", matcher.Expected)
}
