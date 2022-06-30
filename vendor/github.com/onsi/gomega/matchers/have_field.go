package matchers

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/onsi/gomega/format"
)

func extractField(actual interface{}, field string) (interface{}, error) {
	fields := strings.SplitN(field, ".", 2)
	actualValue := reflect.ValueOf(actual)

	if actualValue.Kind() == reflect.Ptr {
		actualValue = actualValue.Elem()
	}
	if actualValue == (reflect.Value{}) {
		return nil, fmt.Errorf("HaveField encountered nil while dereferencing a pointer of type %T.", actual)
	}

	if actualValue.Kind() != reflect.Struct {
		return nil, fmt.Errorf("HaveField encountered:\n%s\nWhich is not a struct.", format.Object(actual, 1))
	}

	var extractedValue reflect.Value

	if strings.HasSuffix(fields[0], "()") {
		extractedValue = actualValue.MethodByName(strings.TrimSuffix(fields[0], "()"))
		if extractedValue == (reflect.Value{}) {
			return nil, fmt.Errorf("HaveField could not find method named '%s' in struct of type %T.", fields[0], actual)
		}
		t := extractedValue.Type()
		if t.NumIn() != 0 || t.NumOut() != 1 {
			return nil, fmt.Errorf("HaveField found an invalid method named '%s' in struct of type %T.\nMethods must take no arguments and return exactly one value.", fields[0], actual)
		}
		extractedValue = extractedValue.Call([]reflect.Value{})[0]
	} else {
		extractedValue = actualValue.FieldByName(fields[0])
		if extractedValue == (reflect.Value{}) {
			return nil, fmt.Errorf("HaveField could not find field named '%s' in struct:\n%s", fields[0], format.Object(actual, 1))
		}
	}

	if len(fields) == 1 {
		return extractedValue.Interface(), nil
	} else {
		return extractField(extractedValue.Interface(), fields[1])
	}
}

type HaveFieldMatcher struct {
	Field    string
	Expected interface{}

	extractedField  interface{}
	expectedMatcher omegaMatcher
}

func (matcher *HaveFieldMatcher) Match(actual interface{}) (success bool, err error) {
	matcher.extractedField, err = extractField(actual, matcher.Field)
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
