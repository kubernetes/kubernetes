/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package matchers

import (
	"errors"
	"fmt"
	"reflect"
	"runtime/debug"
	"strings"

	errorsutil "k8s.io/kubernetes/pkg/util/errors"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/types"
)

type StructMatcher struct {
	// Matchers for each field.
	Fields Fields
	// Whether extra fields are considered an error.
	Strict bool

	// State.
	failures []error
}

// Field name to matcher.
type Fields map[string]types.GomegaMatcher

func (m *StructMatcher) Match(actual interface{}) (success bool, err error) {
	if reflect.TypeOf(actual).Kind() != reflect.Struct {
		return false, fmt.Errorf("%v is type %T, expected struct", actual, actual)
	}

	m.failures = m.matchFields(actual)
	if len(m.failures) > 0 {
		return false, nil
	}
	return true, nil
}

func (m *StructMatcher) matchFields(actual interface{}) (errs []error) {
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
				if m.Strict {
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
				if nesting, ok := matcher.(NestingMatcher); ok {
					return errorsutil.NewAggregate(nesting.Failures())
				}
				return errors.New(matcher.FailureMessage(field))
			}
			return nil
		}()
		if err != nil {
			errs = append(errs, Nest("."+fieldName, err))
		}
	}

	for field := range m.Fields {
		if !fields[field] {
			errs = append(errs, fmt.Errorf("missing expected field %s", field))
		}
	}

	return errs
}

func (m *StructMatcher) FailureMessage(actual interface{}) (message string) {
	failures := make([]string, len(m.failures))
	for i := range m.failures {
		failures[i] = m.failures[i].Error()
	}
	return format.Message(reflect.TypeOf(actual).Name(),
		fmt.Sprintf("to match struct matcher: {\n%v\n}\n", strings.Join(failures, "\n")))
}

func (m *StructMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to match struct matcher")
}

func (m *StructMatcher) Failures() []error {
	return m.failures
}
