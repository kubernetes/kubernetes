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

func MatchAllKeys(keys Keys) types.GomegaMatcher {
	return &KeysMatcher{
		Keys: keys,
	}
}

func MatchKeys(options Options, keys Keys) types.GomegaMatcher {
	return &KeysMatcher{
		Keys:          keys,
		IgnoreExtras:  options&IgnoreExtras != 0,
		IgnoreMissing: options&IgnoreMissing != 0,
	}
}

type KeysMatcher struct {
	// Matchers for each key.
	Keys Keys

	// Whether to ignore extra keys or consider it an error.
	IgnoreExtras bool
	// Whether to ignore missing keys or consider it an error.
	IgnoreMissing bool

	// State.
	failures []error
}

type Keys map[interface{}]types.GomegaMatcher

func (m *KeysMatcher) Match(actual interface{}) (success bool, err error) {
	if reflect.TypeOf(actual).Kind() != reflect.Map {
		return false, fmt.Errorf("%v is type %T, expected map", actual, actual)
	}

	m.failures = m.matchKeys(actual)
	if len(m.failures) > 0 {
		return false, nil
	}
	return true, nil
}

func (m *KeysMatcher) matchKeys(actual interface{}) (errs []error) {
	actualValue := reflect.ValueOf(actual)
	keys := map[interface{}]bool{}
	for _, keyValue := range actualValue.MapKeys() {
		key := keyValue.Interface()
		keys[key] = true

		err := func() (err error) {
			// This test relies heavily on reflect, which tends to panic.
			// Recover here to provide more useful error messages in that case.
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("panic checking %+v: %v\n%s", actual, r, debug.Stack())
				}
			}()

			matcher, ok := m.Keys[key]
			if !ok {
				if !m.IgnoreExtras {
					return fmt.Errorf("unexpected key %s: %+v", key, actual)
				}
				return nil
			}

			valInterface := actualValue.MapIndex(keyValue).Interface()

			match, err := matcher.Match(valInterface)
			if err != nil {
				return err
			}

			if !match {
				if nesting, ok := matcher.(errorsutil.NestingMatcher); ok {
					return errorsutil.AggregateError(nesting.Failures())
				}
				return errors.New(matcher.FailureMessage(valInterface))
			}
			return nil
		}()
		if err != nil {
			errs = append(errs, errorsutil.Nest(fmt.Sprintf(".%#v", key), err))
		}
	}

	for key := range m.Keys {
		if !keys[key] && !m.IgnoreMissing {
			errs = append(errs, fmt.Errorf("missing expected key %s", key))
		}
	}

	return errs
}

func (m *KeysMatcher) FailureMessage(actual interface{}) (message string) {
	failures := make([]string, len(m.failures))
	for i := range m.failures {
		failures[i] = m.failures[i].Error()
	}
	return format.Message(reflect.TypeOf(actual).Name(),
		fmt.Sprintf("to match keys: {\n%v\n}\n", strings.Join(failures, "\n")))
}

func (m *KeysMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to match keys")
}

func (m *KeysMatcher) Failures() []error {
	return m.failures
}
