package gstruct

import (
	"errors"
	"fmt"
	"reflect"
	"runtime/debug"
	"strings"

	"github.com/onsi/gomega/format"
	errorsutil "github.com/onsi/gomega/gstruct/errors"
	"github.com/onsi/gomega/types"
)

//MatchAllFields succeeds if every field of a struct matches the field matcher associated with
//it, and every element matcher is matched.
//  Expect([]string{"a", "b"}).To(MatchAllFields(idFn, gstruct.Fields{
//      "a": BeEqual("a"),
//      "b": BeEqual("b"),
//  })
func MatchAllFields(fields Fields) types.GomegaMatcher {
	return &FieldsMatcher{
		Fields: fields,
	}
}

//MatchFields succeeds if each element of a struct matches the field matcher associated with
//it. It can ignore extra fields and/or missing fields.
//  Expect([]string{"a", "c"}).To(MatchFields(idFn, IgnoreMissing|IgnoreExtra, gstruct.Fields{
//      "a": BeEqual("a")
//      "b": BeEqual("b"),
//  })
func MatchFields(options Options, fields Fields) types.GomegaMatcher {
	return &FieldsMatcher{
		Fields:        fields,
		IgnoreExtras:  options&IgnoreExtras != 0,
		IgnoreMissing: options&IgnoreMissing != 0,
	}
}

type FieldsMatcher struct {
	// Matchers for each field.
	Fields Fields

	// Whether to ignore extra elements or consider it an error.
	IgnoreExtras bool
	// Whether to ignore missing elements or consider it an error.
	IgnoreMissing bool

	// State.
	failures []error
}

// Field name to matcher.
type Fields map[string]types.GomegaMatcher

func (m *FieldsMatcher) Match(actual interface{}) (success bool, err error) {
	if reflect.TypeOf(actual).Kind() != reflect.Struct {
		return false, fmt.Errorf("%v is type %T, expected struct", actual, actual)
	}

	m.failures = m.matchFields(actual)
	if len(m.failures) > 0 {
		return false, nil
	}
	return true, nil
}

func (m *FieldsMatcher) matchFields(actual interface{}) (errs []error) {
	val := reflect.ValueOf(actual)
	typ := val.Type()
	fields := map[string]bool{}
	for i := 0; i < val.NumField(); i++ {
		fieldName := typ.Field(i).Name
		fields[fieldName] = true

		err := func() (err error) {
			// This test relies heavily on reflect, which tends to panic.
			// Recover here to provide more useful error messages in that case.
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("panic checking %+v: %v\n%s", actual, r, debug.Stack())
				}
			}()

			matcher, expected := m.Fields[fieldName]
			if !expected {
				if !m.IgnoreExtras {
					return fmt.Errorf("unexpected field %s: %+v", fieldName, actual)
				}
				return nil
			}

			var field interface{}
			if val.Field(i).IsValid() {
				field = val.Field(i).Interface()
			} else {
				field = reflect.Zero(typ.Field(i).Type)
			}

			match, err := matcher.Match(field)
			if err != nil {
				return err
			} else if !match {
				if nesting, ok := matcher.(errorsutil.NestingMatcher); ok {
					return errorsutil.AggregateError(nesting.Failures())
				}
				return errors.New(matcher.FailureMessage(field))
			}
			return nil
		}()
		if err != nil {
			errs = append(errs, errorsutil.Nest("."+fieldName, err))
		}
	}

	for field := range m.Fields {
		if !fields[field] && !m.IgnoreMissing {
			errs = append(errs, fmt.Errorf("missing expected field %s", field))
		}
	}

	return errs
}

func (m *FieldsMatcher) FailureMessage(actual interface{}) (message string) {
	failures := make([]string, len(m.failures))
	for i := range m.failures {
		failures[i] = m.failures[i].Error()
	}
	return format.Message(reflect.TypeOf(actual).Name(),
		fmt.Sprintf("to match fields: {\n%v\n}\n", strings.Join(failures, "\n")))
}

func (m *FieldsMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to match fields")
}

func (m *FieldsMatcher) Failures() []error {
	return m.failures
}
