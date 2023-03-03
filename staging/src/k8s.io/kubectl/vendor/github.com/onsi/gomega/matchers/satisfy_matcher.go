package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type SatisfyMatcher struct {
	Predicate interface{}

	// cached type
	predicateArgType reflect.Type
}

func NewSatisfyMatcher(predicate interface{}) *SatisfyMatcher {
	if predicate == nil {
		panic("predicate cannot be nil")
	}
	predicateType := reflect.TypeOf(predicate)
	if predicateType.Kind() != reflect.Func {
		panic("predicate must be a function")
	}
	if predicateType.NumIn() != 1 {
		panic("predicate must have 1 argument")
	}
	if predicateType.NumOut() != 1 || predicateType.Out(0).Kind() != reflect.Bool {
		panic("predicate must return bool")
	}

	return &SatisfyMatcher{
		Predicate:        predicate,
		predicateArgType: predicateType.In(0),
	}
}

func (m *SatisfyMatcher) Match(actual interface{}) (success bool, err error) {
	// prepare a parameter to pass to the predicate
	var param reflect.Value
	if actual != nil && reflect.TypeOf(actual).AssignableTo(m.predicateArgType) {
		// The dynamic type of actual is compatible with the predicate argument.
		param = reflect.ValueOf(actual)

	} else if actual == nil && m.predicateArgType.Kind() == reflect.Interface {
		// The dynamic type of actual is unknown, so there's no way to make its
		// reflect.Value. Create a nil of the predicate argument, which is known.
		param = reflect.Zero(m.predicateArgType)

	} else {
		return false, fmt.Errorf("predicate expects '%s' but we have '%T'", m.predicateArgType, actual)
	}

	// call the predicate with `actual`
	fn := reflect.ValueOf(m.Predicate)
	result := fn.Call([]reflect.Value{param})
	return result[0].Bool(), nil
}

func (m *SatisfyMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to satisfy predicate", m.Predicate)
}

func (m *SatisfyMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to not satisfy predicate", m.Predicate)
}
