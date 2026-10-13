package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/types"
)

type WithTransformMatcher struct {
	// input
	Transform any // must be a function of one parameter that returns one value and an optional error
	Matcher   types.GomegaMatcher

	// cached value
	transformArgType reflect.Type

	// state
	transformedValue any
}

// reflect.Type for error
var errorT = reflect.TypeOf((*error)(nil)).Elem()

func NewWithTransformMatcher(transform any, matcher types.GomegaMatcher) *WithTransformMatcher {
	if transform == nil {
		panic("transform function cannot be nil")
	}
	txType := reflect.TypeOf(transform)
	if txType.NumIn() != 1 {
		panic("transform function must have 1 argument")
	}
	if numout := txType.NumOut(); numout != 1 {
		if numout != 2 || !txType.Out(1).AssignableTo(errorT) {
			panic("transform function must either have 1 return value, or 1 return value plus 1 error value")
		}
	}

	return &WithTransformMatcher{
		Transform:        transform,
		Matcher:          matcher,
		transformArgType: reflect.TypeOf(transform).In(0),
	}
}

func (m *WithTransformMatcher) Match(actual any) (bool, error) {
	// prepare a parameter to pass to the Transform function
	var param reflect.Value
	if actual != nil && reflect.TypeOf(actual).AssignableTo(m.transformArgType) {
		// The dynamic type of actual is compatible with the transform argument.
		param = reflect.ValueOf(actual)

	} else if actual == nil && m.transformArgType.Kind() == reflect.Interface {
		// The dynamic type of actual is unknown, so there's no way to make its
		// reflect.Value. Create a nil of the transform argument, which is known.
		param = reflect.Zero(m.transformArgType)

	} else {
		return false, fmt.Errorf("Transform function expects '%s' but we have '%T'", m.transformArgType, actual)
	}

	// call the Transform function with `actual`
	fn := reflect.ValueOf(m.Transform)
	result := fn.Call([]reflect.Value{param})
	if len(result) == 2 {
		if !result[1].IsNil() {
			return false, fmt.Errorf("Transform function failed: %s", result[1].Interface().(error).Error())
		}
	}
	m.transformedValue = result[0].Interface() // expect exactly one value

	return m.Matcher.Match(m.transformedValue)
}

func (m *WithTransformMatcher) FailureMessage(_ any) (message string) {
	return m.Matcher.FailureMessage(m.transformedValue)
}

func (m *WithTransformMatcher) NegatedFailureMessage(_ any) (message string) {
	return m.Matcher.NegatedFailureMessage(m.transformedValue)
}

func (m *WithTransformMatcher) MatchMayChangeInTheFuture(_ any) bool {
	// TODO: Maybe this should always just return true? (Only an issue for non-deterministic transformers.)
	//
	// Querying the next matcher is fine if the transformer always will return the same value.
	// But if the transformer is non-deterministic and returns a different value each time, then there
	// is no point in querying the next matcher, since it can only comment on the last transformed value.
	return types.MatchMayChangeInTheFuture(m.Matcher, m.transformedValue)
}
