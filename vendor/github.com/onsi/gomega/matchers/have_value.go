package matchers

import (
	"errors"
	"reflect"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/types"
)

const maxIndirections = 31

type HaveValueMatcher struct {
	Matcher        types.GomegaMatcher // the matcher to apply to the "resolved" actual value.
	resolvedActual any                 // the ("resolved") value.
}

func (m *HaveValueMatcher) Match(actual any) (bool, error) {
	val := reflect.ValueOf(actual)
	for allowedIndirs := maxIndirections; allowedIndirs > 0; allowedIndirs-- {
		// return an error if value isn't valid. Please note that we cannot
		// check for nil here, as we might not deal with a pointer or interface
		// at this point.
		if !val.IsValid() {
			return false, errors.New(format.Message(
				actual, "not to be <nil>"))
		}
		switch val.Kind() {
		case reflect.Ptr, reflect.Interface:
			// resolve pointers and interfaces to their values, then rinse and
			// repeat.
			if val.IsNil() {
				return false, errors.New(format.Message(
					actual, "not to be <nil>"))
			}
			val = val.Elem()
			continue
		default:
			// forward the final value to the specified matcher.
			m.resolvedActual = val.Interface()
			return m.Matcher.Match(m.resolvedActual)
		}
	}
	// too many indirections: extreme star gazing, indeed...?
	return false, errors.New(format.Message(actual, "too many indirections"))
}

func (m *HaveValueMatcher) FailureMessage(_ any) (message string) {
	return m.Matcher.FailureMessage(m.resolvedActual)
}

func (m *HaveValueMatcher) NegatedFailureMessage(_ any) (message string) {
	return m.Matcher.NegatedFailureMessage(m.resolvedActual)
}
