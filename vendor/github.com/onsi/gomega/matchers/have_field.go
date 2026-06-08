package matchers

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/onsi/gomega/format"
)

// missingFieldError represents a missing field extraction error that
// HaveExistingFieldMatcher can ignore, as opposed to other, sever field
// extraction errors, such as nil pointers, et cetera.
type missingFieldError string

func (e missingFieldError) Error() string {
	return string(e)
}

func extractField(actual any, field string, matchername string) (any, error) {
	fields := strings.SplitN(field, ".", 2)
	actualValue := reflect.ValueOf(actual)

	if actualValue.Kind() == reflect.Ptr {
		actualValue = actualValue.Elem()
	}
	if actualValue == (reflect.Value{}) {
		return nil, fmt.Errorf("%s encountered nil while dereferencing a pointer of type %T.", matchername, actual)
	}

	if actualValue.Kind() != reflect.Struct {
		return nil, fmt.Errorf("%s encountered:\n%s\nWhich is not a struct.", matchername, format.Object(actual, 1))
	}

	var extractedValue reflect.Value

	if strings.HasSuffix(fields[0], "()") {
		extractedValue = actualValue.MethodByName(strings.TrimSuffix(fields[0], "()"))
		if extractedValue == (reflect.Value{}) && actualValue.CanAddr() {
			extractedValue = actualValue.Addr().MethodByName(strings.TrimSuffix(fields[0], "()"))
		}
		if extractedValue == (reflect.Value{}) {
			ptr := reflect.New(actualValue.Type())
			ptr.Elem().Set(actualValue)
			extractedValue = ptr.MethodByName(strings.TrimSuffix(fields[0], "()"))
			if extractedValue == (reflect.Value{}) {
				return nil, missingFieldError(fmt.Sprintf("%s could not find method named '%s' in struct of type %T.", matchername, fields[0], actual))
			}
		}
		t := extractedValue.Type()
		if t.NumIn() != 0 || t.NumOut() != 1 {
			return nil, fmt.Errorf("%s found an invalid method named '%s' in struct of type %T.\nMethods must take no arguments and return exactly one value.", matchername, fields[0], actual)
		}
		extractedValue = extractedValue.Call([]reflect.Value{})[0]
	} else {
		extractedValue = actualValue.FieldByName(fields[0])
		if extractedValue == (reflect.Value{}) {
			return nil, missingFieldError(fmt.Sprintf("%s could not find field named '%s' in struct:\n%s", matchername, fields[0], format.Object(actual, 1)))
		}
	}

	if len(fields) == 1 {
		return extractedValue.Interface(), nil
	} else {
		return extractField(extractedValue.Interface(), fields[1], matchername)
	}
}

type HaveFieldMatcher struct {
	Field    string
	Expected any
}

func (matcher *HaveFieldMatcher) expectedMatcher() omegaMatcher {
	var isMatcher bool
	expectedMatcher, isMatcher := matcher.Expected.(omegaMatcher)
	if !isMatcher {
		expectedMatcher = &EqualMatcher{Expected: matcher.Expected}
	}
	return expectedMatcher
}

func (matcher *HaveFieldMatcher) Match(actual any) (success bool, err error) {
	extractedField, err := extractField(actual, matcher.Field, "HaveField")
	if err != nil {
		return false, err
	}

	return matcher.expectedMatcher().Match(extractedField)
}

func (matcher *HaveFieldMatcher) FailureMessage(actual any) (message string) {
	extractedField, err := extractField(actual, matcher.Field, "HaveField")
	if err != nil {
		// this really shouldn't happen
		return fmt.Sprintf("Failed to extract field '%s': %s", matcher.Field, err)
	}
	message = fmt.Sprintf("Value for field '%s' failed to satisfy matcher.\n", matcher.Field)
	message += matcher.expectedMatcher().FailureMessage(extractedField)

	return message
}

func (matcher *HaveFieldMatcher) NegatedFailureMessage(actual any) (message string) {
	extractedField, err := extractField(actual, matcher.Field, "HaveField")
	if err != nil {
		// this really shouldn't happen
		return fmt.Sprintf("Failed to extract field '%s': %s", matcher.Field, err)
	}
	message = fmt.Sprintf("Value for field '%s' satisfied matcher, but should not have.\n", matcher.Field)
	message += matcher.expectedMatcher().NegatedFailureMessage(extractedField)

	return message
}
