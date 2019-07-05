package gstruct

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/types"
)

//PointTo applies the given matcher to the value pointed to by actual. It fails if the pointer is
//nil.
//  actual := 5
//  Expect(&actual).To(PointTo(Equal(5)))
func PointTo(matcher types.GomegaMatcher) types.GomegaMatcher {
	return &PointerMatcher{
		Matcher: matcher,
	}
}

type PointerMatcher struct {
	Matcher types.GomegaMatcher

	// Failure message.
	failure string
}

func (m *PointerMatcher) Match(actual interface{}) (bool, error) {
	val := reflect.ValueOf(actual)

	// return error if actual type is not a pointer
	if val.Kind() != reflect.Ptr {
		return false, fmt.Errorf("PointerMatcher expects a pointer but we have '%s'", val.Kind())
	}

	if !val.IsValid() || val.IsNil() {
		m.failure = format.Message(actual, "not to be <nil>")
		return false, nil
	}

	// Forward the value.
	elem := val.Elem().Interface()
	match, err := m.Matcher.Match(elem)
	if !match {
		m.failure = m.Matcher.FailureMessage(elem)
	}
	return match, err
}

func (m *PointerMatcher) FailureMessage(_ interface{}) (message string) {
	return m.failure
}

func (m *PointerMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return m.Matcher.NegatedFailureMessage(actual)
}
