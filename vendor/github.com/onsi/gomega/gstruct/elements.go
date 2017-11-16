package gstruct

import (
	"errors"
	"fmt"
	"reflect"
	"runtime/debug"

	"github.com/onsi/gomega/format"
	errorsutil "github.com/onsi/gomega/gstruct/errors"
	"github.com/onsi/gomega/types"
)

//MatchAllElements succeeds if every element of a slice matches the element matcher it maps to
//through the id function, and every element matcher is matched.
//  Expect([]string{"a", "b"}).To(MatchAllElements(idFn, matchers.Elements{
//      "a": BeEqual("a"),
//      "b": BeEqual("b"),
//  })
func MatchAllElements(identifier Identifier, elements Elements) types.GomegaMatcher {
	return &ElementsMatcher{
		Identifier: identifier,
		Elements:   elements,
	}
}

//MatchElements succeeds if each element of a slice matches the element matcher it maps to
//through the id function. It can ignore extra elements and/or missing elements.
//  Expect([]string{"a", "c"}).To(MatchElements(idFn, IgnoreMissing|IgnoreExtra, matchers.Elements{
//      "a": BeEqual("a")
//      "b": BeEqual("b"),
//  })
func MatchElements(identifier Identifier, options Options, elements Elements) types.GomegaMatcher {
	return &ElementsMatcher{
		Identifier:    identifier,
		Elements:      elements,
		IgnoreExtras:  options&IgnoreExtras != 0,
		IgnoreMissing: options&IgnoreMissing != 0,
	}
}

// ElementsMatcher is a NestingMatcher that applies custom matchers to each element of a slice mapped
// by the Identifier function.
// TODO: Extend this to work with arrays & maps (map the key) as well.
type ElementsMatcher struct {
	// Matchers for each element.
	Elements Elements
	// Function mapping an element to the string key identifying its matcher.
	Identifier Identifier

	// Whether to ignore extra elements or consider it an error.
	IgnoreExtras bool
	// Whether to ignore missing elements or consider it an error.
	IgnoreMissing bool

	// State.
	failures []error
}

// Element ID to matcher.
type Elements map[string]types.GomegaMatcher

// Function for identifying (mapping) elements.
type Identifier func(element interface{}) string

func (m *ElementsMatcher) Match(actual interface{}) (success bool, err error) {
	if reflect.TypeOf(actual).Kind() != reflect.Slice {
		return false, fmt.Errorf("%v is type %T, expected slice", actual, actual)
	}

	m.failures = m.matchElements(actual)
	if len(m.failures) > 0 {
		return false, nil
	}
	return true, nil
}

func (m *ElementsMatcher) matchElements(actual interface{}) (errs []error) {
	// Provide more useful error messages in the case of a panic.
	defer func() {
		if err := recover(); err != nil {
			errs = append(errs, fmt.Errorf("panic checking %+v: %v\n%s", actual, err, debug.Stack()))
		}
	}()

	val := reflect.ValueOf(actual)
	elements := map[string]bool{}
	for i := 0; i < val.Len(); i++ {
		element := val.Index(i).Interface()
		id := m.Identifier(element)
		// TODO: Add options to ignore & match duplicates.
		if elements[id] {
			errs = append(errs, fmt.Errorf("found duplicate element ID %s", id))
			continue
		}
		elements[id] = true

		matcher, expected := m.Elements[id]
		if !expected {
			if !m.IgnoreExtras {
				errs = append(errs, fmt.Errorf("unexpected element %s", id))
			}
			continue
		}

		match, err := matcher.Match(element)
		if match {
			continue
		}

		if err == nil {
			if nesting, ok := matcher.(errorsutil.NestingMatcher); ok {
				err = errorsutil.AggregateError(nesting.Failures())
			} else {
				err = errors.New(matcher.FailureMessage(element))
			}
		}
		errs = append(errs, errorsutil.Nest(fmt.Sprintf("[%s]", id), err))
	}

	for id := range m.Elements {
		if !elements[id] && !m.IgnoreMissing {
			errs = append(errs, fmt.Errorf("missing expected element %s", id))
		}
	}

	return errs
}

func (m *ElementsMatcher) FailureMessage(actual interface{}) (message string) {
	failure := errorsutil.AggregateError(m.failures)
	return format.Message(actual, fmt.Sprintf("to match elements: %v", failure))
}

func (m *ElementsMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to match elements")
}

func (m *ElementsMatcher) Failures() []error {
	return m.failures
}
