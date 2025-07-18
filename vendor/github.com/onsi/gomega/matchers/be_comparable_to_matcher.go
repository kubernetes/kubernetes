package matchers

import (
	"bytes"
	"fmt"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/gomega/format"
)

type BeComparableToMatcher struct {
	Expected any
	Options  cmp.Options
}

func (matcher *BeComparableToMatcher) Match(actual any) (success bool, matchErr error) {
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

	defer func() {
		if r := recover(); r != nil {
			success = false
			if err, ok := r.(error); ok {
				matchErr = err
			} else if errMsg, ok := r.(string); ok {
				matchErr = fmt.Errorf(errMsg)
			}
		}
	}()

	return cmp.Equal(actual, matcher.Expected, matcher.Options...), nil
}

func (matcher *BeComparableToMatcher) FailureMessage(actual any) (message string) {
	return fmt.Sprint("Expected object to be comparable, diff: ", cmp.Diff(actual, matcher.Expected, matcher.Options...))
}

func (matcher *BeComparableToMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to be comparable to", matcher.Expected)
}
