package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type PanicMatcher struct {
	Expected interface{}
	object   interface{}
}

func (matcher *PanicMatcher) Match(actual interface{}) (success bool, err error) {
	if actual == nil {
		return false, fmt.Errorf("PanicMatcher expects a non-nil actual.")
	}

	actualType := reflect.TypeOf(actual)
	if actualType.Kind() != reflect.Func {
		return false, fmt.Errorf("PanicMatcher expects a function.  Got:\n%s", format.Object(actual, 1))
	}
	if !(actualType.NumIn() == 0 && actualType.NumOut() == 0) {
		return false, fmt.Errorf("PanicMatcher expects a function with no arguments and no return value.  Got:\n%s", format.Object(actual, 1))
	}

	success = false
	defer func() {
		if e := recover(); e != nil {
			matcher.object = e

			if matcher.Expected == nil {
				success = true
				return
			}

			valueMatcher, valueIsMatcher := matcher.Expected.(omegaMatcher)
			if !valueIsMatcher {
				valueMatcher = &EqualMatcher{Expected: matcher.Expected}
			}

			success, err = valueMatcher.Match(e)
			if err != nil {
				err = fmt.Errorf("PanicMatcher's value matcher failed with:\n%s%s", format.Indent, err.Error())
			}
		}
	}()

	reflect.ValueOf(actual).Call([]reflect.Value{})

	return
}

func (matcher *PanicMatcher) FailureMessage(actual interface{}) (message string) {
	if matcher.Expected == nil {
		// We wanted any panic to occur, but none did.
		return format.Message(actual, "to panic")
	}

	if matcher.object == nil {
		// We wanted a panic with a specific value to occur, but none did.
		switch matcher.Expected.(type) {
		case omegaMatcher:
			return format.Message(actual, "to panic with a value matching", matcher.Expected)
		default:
			return format.Message(actual, "to panic with", matcher.Expected)
		}
	}

	// We got a panic, but the value isn't what we expected.
	switch matcher.Expected.(type) {
	case omegaMatcher:
		return format.Message(
			actual,
			fmt.Sprintf(
				"to panic with a value matching\n%s\nbut panicked with\n%s",
				format.Object(matcher.Expected, 1),
				format.Object(matcher.object, 1),
			),
		)
	default:
		return format.Message(
			actual,
			fmt.Sprintf(
				"to panic with\n%s\nbut panicked with\n%s",
				format.Object(matcher.Expected, 1),
				format.Object(matcher.object, 1),
			),
		)
	}
}

func (matcher *PanicMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	if matcher.Expected == nil {
		// We didn't want any panic to occur, but one did.
		return format.Message(actual, fmt.Sprintf("not to panic, but panicked with\n%s", format.Object(matcher.object, 1)))
	}

	// We wanted a to ensure a panic with a specific value did not occur, but it did.
	switch matcher.Expected.(type) {
	case omegaMatcher:
		return format.Message(
			actual,
			fmt.Sprintf(
				"not to panic with a value matching\n%s\nbut panicked with\n%s",
				format.Object(matcher.Expected, 1),
				format.Object(matcher.object, 1),
			),
		)
	default:
		return format.Message(actual, "not to panic with", matcher.Expected)
	}
}
