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

func extractField(actual interface{}, field string, matchername string) (interface{}, error) {
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
			return nil, missingFieldError(fmt.Sprintf("%s could not find method named '%s' in struct of type %T.", matchername, fields[0], actual))
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
	Expected interface{}

	extractedField  interface{}
	expectedMatcher omegaMatcher
}

func (matcher *HaveFieldMatcher) Match(actual interface{}) (success bool, err error) {
	matcher.extractedField, err = extractField(actual, matcher.Field, "HaveField")
	if err != nil {
		return false, err
	}

	var isMatcher bool
	matcher.expectedMatcher, isMatcher = matcher.Expected.(omegaMatcher)
	if !isMatcher {
		matcher.expectedMatcher = &EqualMatcher{Expected: matcher.Expected}
	}

	return matcher.expectedMatcher.Match(matcher.extractedField)
}

func (matcher *HaveFieldMatcher) FailureMessage(actual interface{}) (message string) {
	message = fmt.Sprintf("Value for field '%s' failed to satisfy matcher.\n", matcher.Field)
	message += matcher.expectedMatcher.FailureMessage(matcher.extractedField)

	return message
}

func (matcher *HaveFieldMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	message = fmt.Sprintf("Value for field '%s' satisfied matcher, but should not have.\n", matcher.Field)
	message += matcher.expectedMatcher.NegatedFailureMessage(matcher.extractedField)

	return message
}
