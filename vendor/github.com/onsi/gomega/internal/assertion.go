package internal

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/types"
)

type Assertion struct {
	actuals     []interface{} // actual value plus all extra values
	actualIndex int           // value to pass to the matcher
	vet         vetinari      // the vet to call before calling Gomega matcher
	offset      int
	g           *Gomega
}

// ...obligatory discworld reference, as "vetineer" doesn't sound ... quite right.
type vetinari func(assertion *Assertion, optionalDescription ...interface{}) bool

func NewAssertion(actualInput interface{}, g *Gomega, offset int, extra ...interface{}) *Assertion {
	return &Assertion{
		actuals:     append([]interface{}{actualInput}, extra...),
		actualIndex: 0,
		vet:         (*Assertion).vetActuals,
		offset:      offset,
		g:           g,
	}
}

func (assertion *Assertion) WithOffset(offset int) types.Assertion {
	assertion.offset = offset
	return assertion
}

func (assertion *Assertion) Error() types.Assertion {
	return &Assertion{
		actuals:     assertion.actuals,
		actualIndex: len(assertion.actuals) - 1,
		vet:         (*Assertion).vetError,
		offset:      assertion.offset,
		g:           assertion.g,
	}
}

func (assertion *Assertion) Should(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.g.THelper()
	return assertion.vet(assertion, optionalDescription...) && assertion.match(matcher, true, optionalDescription...)
}

func (assertion *Assertion) ShouldNot(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.g.THelper()
	return assertion.vet(assertion, optionalDescription...) && assertion.match(matcher, false, optionalDescription...)
}

func (assertion *Assertion) To(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.g.THelper()
	return assertion.vet(assertion, optionalDescription...) && assertion.match(matcher, true, optionalDescription...)
}

func (assertion *Assertion) ToNot(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.g.THelper()
	return assertion.vet(assertion, optionalDescription...) && assertion.match(matcher, false, optionalDescription...)
}

func (assertion *Assertion) NotTo(matcher types.GomegaMatcher, optionalDescription ...interface{}) bool {
	assertion.g.THelper()
	return assertion.vet(assertion, optionalDescription...) && assertion.match(matcher, false, optionalDescription...)
}

func (assertion *Assertion) buildDescription(optionalDescription ...interface{}) string {
	switch len(optionalDescription) {
	case 0:
		return ""
	case 1:
		if describe, ok := optionalDescription[0].(func() string); ok {
			return describe() + "\n"
		}
	}
	return fmt.Sprintf(optionalDescription[0].(string), optionalDescription[1:]...) + "\n"
}

func (assertion *Assertion) match(matcher types.GomegaMatcher, desiredMatch bool, optionalDescription ...interface{}) bool {
	actualInput := assertion.actuals[assertion.actualIndex]
	matches, err := matcher.Match(actualInput)
	assertion.g.THelper()
	if err != nil {
		description := assertion.buildDescription(optionalDescription...)
		assertion.g.Fail(description+err.Error(), 2+assertion.offset)
		return false
	}
	if matches != desiredMatch {
		var message string
		if desiredMatch {
			message = matcher.FailureMessage(actualInput)
		} else {
			message = matcher.NegatedFailureMessage(actualInput)
		}
		description := assertion.buildDescription(optionalDescription...)
		assertion.g.Fail(description+message, 2+assertion.offset)
		return false
	}

	return true
}

// vetActuals vets the actual values, with the (optional) exception of a
// specific value, such as the first value in case non-error assertions, or the
// last value in case of Error()-based assertions.
func (assertion *Assertion) vetActuals(optionalDescription ...interface{}) bool {
	success, message := vetActuals(assertion.actuals, assertion.actualIndex)
	if success {
		return true
	}

	description := assertion.buildDescription(optionalDescription...)
	assertion.g.THelper()
	assertion.g.Fail(description+message, 2+assertion.offset)
	return false
}

// vetError vets the actual values, except for the final error value, in case
// the final error value is non-zero. Otherwise, it doesn't vet the actual
// values, as these are allowed to take on any values unless there is a non-zero
// error value.
func (assertion *Assertion) vetError(optionalDescription ...interface{}) bool {
	if err := assertion.actuals[assertion.actualIndex]; err != nil {
		// Go error result idiom: all other actual values must be zero values.
		return assertion.vetActuals(optionalDescription...)
	}
	return true
}

// vetActuals vets a slice of actual values, optionally skipping a particular
// value slice element, such as the first or last value slice element.
func vetActuals(actuals []interface{}, skipIndex int) (bool, string) {
	for i, actual := range actuals {
		if i == skipIndex {
			continue
		}
		if actual != nil {
			zeroValue := reflect.Zero(reflect.TypeOf(actual)).Interface()
			if !reflect.DeepEqual(zeroValue, actual) {
				message := fmt.Sprintf("Unexpected non-nil/non-zero argument at index %d:\n\t<%T>: %#v", i, actual, actual)
				return false, message
			}
		}
	}
	return true, ""
}
