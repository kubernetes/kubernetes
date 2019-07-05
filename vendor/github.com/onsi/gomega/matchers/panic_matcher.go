package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type PanicMatcher struct {
	object interface{}
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
			success = true
		}
	}()

	reflect.ValueOf(actual).Call([]reflect.Value{})

	return
}

func (matcher *PanicMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to panic")
}

func (matcher *PanicMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, fmt.Sprintf("not to panic, but panicked with\n%s", format.Object(matcher.object, 1)))
}
