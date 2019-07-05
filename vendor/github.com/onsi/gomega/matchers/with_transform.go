package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/internal/oraclematcher"
	"github.com/onsi/gomega/types"
)

type WithTransformMatcher struct {
	// input
	Transform interface{} // must be a function of one parameter that returns one value
	Matcher   types.GomegaMatcher

	// cached value
	transformArgType reflect.Type

	// state
	transformedValue interface{}
}

func NewWithTransformMatcher(transform interface{}, matcher types.GomegaMatcher) *WithTransformMatcher {
	if transform == nil {
		panic("transform function cannot be nil")
	}
	txType := reflect.TypeOf(transform)
	if txType.NumIn() != 1 {
		panic("transform function must have 1 argument")
	}
	if txType.NumOut() != 1 {
		panic("transform function must have 1 return value")
	}

	return &WithTransformMatcher{
		Transform:        transform,
		Matcher:          matcher,
		transformArgType: reflect.TypeOf(transform).In(0),
	}
}

func (m *WithTransformMatcher) Match(actual interface{}) (bool, error) {
	// return error if actual's type is incompatible with Transform function's argument type
	actualType := reflect.TypeOf(actual)
	if !actualType.AssignableTo(m.transformArgType) {
		return false, fmt.Errorf("Transform function expects '%s' but we have '%s'", m.transformArgType, actualType)
	}

	// call the Transform function with `actual`
	fn := reflect.ValueOf(m.Transform)
	result := fn.Call([]reflect.Value{reflect.ValueOf(actual)})
	m.transformedValue = result[0].Interface() // expect exactly one value

	return m.Matcher.Match(m.transformedValue)
}

func (m *WithTransformMatcher) FailureMessage(_ interface{}) (message string) {
	return m.Matcher.FailureMessage(m.transformedValue)
}

func (m *WithTransformMatcher) NegatedFailureMessage(_ interface{}) (message string) {
	return m.Matcher.NegatedFailureMessage(m.transformedValue)
}

func (m *WithTransformMatcher) MatchMayChangeInTheFuture(_ interface{}) bool {
	// TODO: Maybe this should always just return true? (Only an issue for non-deterministic transformers.)
	//
	// Querying the next matcher is fine if the transformer always will return the same value.
	// But if the transformer is non-deterministic and returns a different value each time, then there
	// is no point in querying the next matcher, since it can only comment on the last transformed value.
	return oraclematcher.MatchMayChangeInTheFuture(m.Matcher, m.transformedValue)
}
