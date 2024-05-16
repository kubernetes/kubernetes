// untested sections: 2

package matchers

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
)

type ContainElementMatcher struct {
	Element interface{}
	Result  []interface{}
}

func (matcher *ContainElementMatcher) Match(actual interface{}) (success bool, err error) {
	if !isArrayOrSlice(actual) && !isMap(actual) {
		return false, fmt.Errorf("ContainElement matcher expects an array/slice/map.  Got:\n%s", format.Object(actual, 1))
	}

	var actualT reflect.Type
	var result reflect.Value
	switch l := len(matcher.Result); {
	case l > 1:
		return false, errors.New("ContainElement matcher expects at most a single optional pointer to store its findings at")
	case l == 1:
		if reflect.ValueOf(matcher.Result[0]).Kind() != reflect.Ptr {
			return false, fmt.Errorf("ContainElement matcher expects a non-nil pointer to store its findings at.  Got\n%s",
				format.Object(matcher.Result[0], 1))
		}
		actualT = reflect.TypeOf(actual)
		resultReference := matcher.Result[0]
		result = reflect.ValueOf(resultReference).Elem() // what ResultReference points to, to stash away our findings
		switch result.Kind() {
		case reflect.Array:
			return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
				reflect.SliceOf(actualT.Elem()).String(), result.Type().String())
		case reflect.Slice:
			if !isArrayOrSlice(actual) {
				return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
					reflect.MapOf(actualT.Key(), actualT.Elem()).String(), result.Type().String())
			}
			if !actualT.Elem().AssignableTo(result.Type().Elem()) {
				return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
					actualT.String(), result.Type().String())
			}
		case reflect.Map:
			if !isMap(actual) {
				return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
					actualT.String(), result.Type().String())
			}
			if !actualT.AssignableTo(result.Type()) {
				return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
					actualT.String(), result.Type().String())
			}
		default:
			if !actualT.Elem().AssignableTo(result.Type()) {
				return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
					actualT.Elem().String(), result.Type().String())
			}
		}
	}

	elemMatcher, elementIsMatcher := matcher.Element.(omegaMatcher)
	if !elementIsMatcher {
		elemMatcher = &EqualMatcher{Expected: matcher.Element}
	}

	value := reflect.ValueOf(actual)
	var valueAt func(int) interface{}

	var getFindings func() reflect.Value
	var foundAt func(int)

	if isMap(actual) {
		keys := value.MapKeys()
		valueAt = func(i int) interface{} {
			return value.MapIndex(keys[i]).Interface()
		}
		if result.Kind() != reflect.Invalid {
			fm := reflect.MakeMap(actualT)
			getFindings = func() reflect.Value {
				return fm
			}
			foundAt = func(i int) {
				fm.SetMapIndex(keys[i], value.MapIndex(keys[i]))
			}
		}
	} else {
		valueAt = func(i int) interface{} {
			return value.Index(i).Interface()
		}
		if result.Kind() != reflect.Invalid {
			var f reflect.Value
			if result.Kind() == reflect.Slice {
				f = reflect.MakeSlice(result.Type(), 0, 0)
			} else {
				f = reflect.MakeSlice(reflect.SliceOf(result.Type()), 0, 0)
			}
			getFindings = func() reflect.Value {
				return f
			}
			foundAt = func(i int) {
				f = reflect.Append(f, value.Index(i))
			}
		}
	}

	var lastError error
	for i := 0; i < value.Len(); i++ {
		elem := valueAt(i)
		success, err := elemMatcher.Match(elem)
		if err != nil {
			lastError = err
			continue
		}
		if success {
			if result.Kind() == reflect.Invalid {
				return true, nil
			}
			foundAt(i)
		}
	}

	// when the expectation isn't interested in the findings except for success
	// or non-success, then we're done here and return the last matcher error
	// seen, if any, as well as non-success.
	if result.Kind() == reflect.Invalid {
		return false, lastError
	}

	// pick up any findings the test is interested in as it specified a non-nil
	// result reference. However, the expection always is that there are at
	// least one or multiple findings. So, if a result is expected, but we had
	// no findings, then this is an error.
	findings := getFindings()
	if findings.Len() == 0 {
		return false, lastError
	}

	// there's just a single finding and the result is neither a slice nor a map
	// (so it's a scalar): pick the one and only finding and return it in the
	// place the reference points to.
	if findings.Len() == 1 && !isArrayOrSlice(result.Interface()) && !isMap(result.Interface()) {
		if isMap(actual) {
			miter := findings.MapRange()
			miter.Next()
			result.Set(miter.Value())
		} else {
			result.Set(findings.Index(0))
		}
		return true, nil
	}

	// at least one or even multiple findings and a the result references a
	// slice or a map, so all we need to do is to store our findings where the
	// reference points to.
	if !findings.Type().AssignableTo(result.Type()) {
		return false, fmt.Errorf("ContainElement cannot return multiple findings.  Need *%s, got *%s",
			findings.Type().String(), result.Type().String())
	}
	result.Set(findings)
	return true, nil
}

func (matcher *ContainElementMatcher) FailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "to contain element matching", matcher.Element)
}

func (matcher *ContainElementMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to contain element matching", matcher.Element)
}
