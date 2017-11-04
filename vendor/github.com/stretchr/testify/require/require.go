/*
* CODE GENERATED AUTOMATICALLY WITH github.com/stretchr/testify/_codegen
* THIS FILE MUST NOT BE EDITED BY HAND
 */

package require

import (
	assert "github.com/stretchr/testify/assert"
	http "net/http"
	url "net/url"
	time "time"
)

// Condition uses a Comparison to assert a complex condition.
func Condition(t TestingT, comp assert.Comparison, msgAndArgs ...interface{}) {
	if !assert.Condition(t, comp, msgAndArgs...) {
		t.FailNow()
	}
}

// Conditionf uses a Comparison to assert a complex condition.
func Conditionf(t TestingT, comp assert.Comparison, msg string, args ...interface{}) {
	if !assert.Conditionf(t, comp, msg, args...) {
		t.FailNow()
	}
}

// Contains asserts that the specified string, list(array, slice...) or map contains the
// specified substring or element.
//
//    assert.Contains(t, "Hello World", "World")
//    assert.Contains(t, ["Hello", "World"], "World")
//    assert.Contains(t, {"Hello": "World"}, "Hello")
//
// Returns whether the assertion was successful (true) or not (false).
func Contains(t TestingT, s interface{}, contains interface{}, msgAndArgs ...interface{}) {
	if !assert.Contains(t, s, contains, msgAndArgs...) {
		t.FailNow()
	}
}

// Containsf asserts that the specified string, list(array, slice...) or map contains the
// specified substring or element.
//
//    assert.Containsf(t, "Hello World", "World", "error message %s", "formatted")
//    assert.Containsf(t, ["Hello", "World"], "World", "error message %s", "formatted")
//    assert.Containsf(t, {"Hello": "World"}, "Hello", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Containsf(t TestingT, s interface{}, contains interface{}, msg string, args ...interface{}) {
	if !assert.Containsf(t, s, contains, msg, args...) {
		t.FailNow()
	}
}

// Empty asserts that the specified object is empty.  I.e. nil, "", false, 0 or either
// a slice or a channel with len == 0.
//
//  assert.Empty(t, obj)
//
// Returns whether the assertion was successful (true) or not (false).
func Empty(t TestingT, object interface{}, msgAndArgs ...interface{}) {
	if !assert.Empty(t, object, msgAndArgs...) {
		t.FailNow()
	}
}

// Emptyf asserts that the specified object is empty.  I.e. nil, "", false, 0 or either
// a slice or a channel with len == 0.
//
//  assert.Emptyf(t, obj, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Emptyf(t TestingT, object interface{}, msg string, args ...interface{}) {
	if !assert.Emptyf(t, object, msg, args...) {
		t.FailNow()
	}
}

// Equal asserts that two objects are equal.
//
//    assert.Equal(t, 123, 123)
//
// Returns whether the assertion was successful (true) or not (false).
//
// Pointer variable equality is determined based on the equality of the
// referenced values (as opposed to the memory addresses). Function equality
// cannot be determined and will always fail.
func Equal(t TestingT, expected interface{}, actual interface{}, msgAndArgs ...interface{}) {
	if !assert.Equal(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}

// EqualError asserts that a function returned an error (i.e. not `nil`)
// and that it is equal to the provided error.
//
//   actualObj, err := SomeFunction()
//   assert.EqualError(t, err,  expectedErrorString)
//
// Returns whether the assertion was successful (true) or not (false).
func EqualError(t TestingT, theError error, errString string, msgAndArgs ...interface{}) {
	if !assert.EqualError(t, theError, errString, msgAndArgs...) {
		t.FailNow()
	}
}

// EqualErrorf asserts that a function returned an error (i.e. not `nil`)
// and that it is equal to the provided error.
//
//   actualObj, err := SomeFunction()
//   assert.EqualErrorf(t, err,  expectedErrorString, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func EqualErrorf(t TestingT, theError error, errString string, msg string, args ...interface{}) {
	if !assert.EqualErrorf(t, theError, errString, msg, args...) {
		t.FailNow()
	}
}

// EqualValues asserts that two objects are equal or convertable to the same types
// and equal.
//
//    assert.EqualValues(t, uint32(123), int32(123))
//
// Returns whether the assertion was successful (true) or not (false).
func EqualValues(t TestingT, expected interface{}, actual interface{}, msgAndArgs ...interface{}) {
	if !assert.EqualValues(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}

// EqualValuesf asserts that two objects are equal or convertable to the same types
// and equal.
//
//    assert.EqualValuesf(t, uint32(123, "error message %s", "formatted"), int32(123))
//
// Returns whether the assertion was successful (true) or not (false).
func EqualValuesf(t TestingT, expected interface{}, actual interface{}, msg string, args ...interface{}) {
	if !assert.EqualValuesf(t, expected, actual, msg, args...) {
		t.FailNow()
	}
}

// Equalf asserts that two objects are equal.
//
//    assert.Equalf(t, 123, 123, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
//
// Pointer variable equality is determined based on the equality of the
// referenced values (as opposed to the memory addresses). Function equality
// cannot be determined and will always fail.
func Equalf(t TestingT, expected interface{}, actual interface{}, msg string, args ...interface{}) {
	if !assert.Equalf(t, expected, actual, msg, args...) {
		t.FailNow()
	}
}

// Error asserts that a function returned an error (i.e. not `nil`).
//
//   actualObj, err := SomeFunction()
//   if assert.Error(t, err) {
// 	   assert.Equal(t, expectedError, err)
//   }
//
// Returns whether the assertion was successful (true) or not (false).
func Error(t TestingT, err error, msgAndArgs ...interface{}) {
	if !assert.Error(t, err, msgAndArgs...) {
		t.FailNow()
	}
}

// Errorf asserts that a function returned an error (i.e. not `nil`).
//
//   actualObj, err := SomeFunction()
//   if assert.Errorf(t, err, "error message %s", "formatted") {
// 	   assert.Equal(t, expectedErrorf, err)
//   }
//
// Returns whether the assertion was successful (true) or not (false).
func Errorf(t TestingT, err error, msg string, args ...interface{}) {
	if !assert.Errorf(t, err, msg, args...) {
		t.FailNow()
	}
}

// Exactly asserts that two objects are equal is value and type.
//
//    assert.Exactly(t, int32(123), int64(123))
//
// Returns whether the assertion was successful (true) or not (false).
func Exactly(t TestingT, expected interface{}, actual interface{}, msgAndArgs ...interface{}) {
	if !assert.Exactly(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}

// Exactlyf asserts that two objects are equal is value and type.
//
//    assert.Exactlyf(t, int32(123, "error message %s", "formatted"), int64(123))
//
// Returns whether the assertion was successful (true) or not (false).
func Exactlyf(t TestingT, expected interface{}, actual interface{}, msg string, args ...interface{}) {
	if !assert.Exactlyf(t, expected, actual, msg, args...) {
		t.FailNow()
	}
}

// Fail reports a failure through
func Fail(t TestingT, failureMessage string, msgAndArgs ...interface{}) {
	if !assert.Fail(t, failureMessage, msgAndArgs...) {
		t.FailNow()
	}
}

// FailNow fails test
func FailNow(t TestingT, failureMessage string, msgAndArgs ...interface{}) {
	if !assert.FailNow(t, failureMessage, msgAndArgs...) {
		t.FailNow()
	}
}

// FailNowf fails test
func FailNowf(t TestingT, failureMessage string, msg string, args ...interface{}) {
	if !assert.FailNowf(t, failureMessage, msg, args...) {
		t.FailNow()
	}
}

// Failf reports a failure through
func Failf(t TestingT, failureMessage string, msg string, args ...interface{}) {
	if !assert.Failf(t, failureMessage, msg, args...) {
		t.FailNow()
	}
}

// False asserts that the specified value is false.
//
//    assert.False(t, myBool)
//
// Returns whether the assertion was successful (true) or not (false).
func False(t TestingT, value bool, msgAndArgs ...interface{}) {
	if !assert.False(t, value, msgAndArgs...) {
		t.FailNow()
	}
}

// Falsef asserts that the specified value is false.
//
//    assert.Falsef(t, myBool, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Falsef(t TestingT, value bool, msg string, args ...interface{}) {
	if !assert.Falsef(t, value, msg, args...) {
		t.FailNow()
	}
}

// HTTPBodyContains asserts that a specified handler returns a
// body that contains a string.
//
//  assert.HTTPBodyContains(t, myHandler, "www.google.com", nil, "I'm Feeling Lucky")
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPBodyContains(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values, str interface{}) {
	if !assert.HTTPBodyContains(t, handler, method, url, values, str) {
		t.FailNow()
	}
}

// HTTPBodyContainsf asserts that a specified handler returns a
// body that contains a string.
//
//  assert.HTTPBodyContainsf(t, myHandler, "www.google.com", nil, "I'm Feeling Lucky", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPBodyContainsf(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values, str interface{}) {
	if !assert.HTTPBodyContainsf(t, handler, method, url, values, str) {
		t.FailNow()
	}
}

// HTTPBodyNotContains asserts that a specified handler returns a
// body that does not contain a string.
//
//  assert.HTTPBodyNotContains(t, myHandler, "www.google.com", nil, "I'm Feeling Lucky")
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPBodyNotContains(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values, str interface{}) {
	if !assert.HTTPBodyNotContains(t, handler, method, url, values, str) {
		t.FailNow()
	}
}

// HTTPBodyNotContainsf asserts that a specified handler returns a
// body that does not contain a string.
//
//  assert.HTTPBodyNotContainsf(t, myHandler, "www.google.com", nil, "I'm Feeling Lucky", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPBodyNotContainsf(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values, str interface{}) {
	if !assert.HTTPBodyNotContainsf(t, handler, method, url, values, str) {
		t.FailNow()
	}
}

// HTTPError asserts that a specified handler returns an error status code.
//
//  assert.HTTPError(t, myHandler, "POST", "/a/b/c", url.Values{"a": []string{"b", "c"}}
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPError(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values) {
	if !assert.HTTPError(t, handler, method, url, values) {
		t.FailNow()
	}
}

// HTTPErrorf asserts that a specified handler returns an error status code.
//
//  assert.HTTPErrorf(t, myHandler, "POST", "/a/b/c", url.Values{"a": []string{"b", "c"}}
//
// Returns whether the assertion was successful (true, "error message %s", "formatted") or not (false).
func HTTPErrorf(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values) {
	if !assert.HTTPErrorf(t, handler, method, url, values) {
		t.FailNow()
	}
}

// HTTPRedirect asserts that a specified handler returns a redirect status code.
//
//  assert.HTTPRedirect(t, myHandler, "GET", "/a/b/c", url.Values{"a": []string{"b", "c"}}
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPRedirect(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values) {
	if !assert.HTTPRedirect(t, handler, method, url, values) {
		t.FailNow()
	}
}

// HTTPRedirectf asserts that a specified handler returns a redirect status code.
//
//  assert.HTTPRedirectf(t, myHandler, "GET", "/a/b/c", url.Values{"a": []string{"b", "c"}}
//
// Returns whether the assertion was successful (true, "error message %s", "formatted") or not (false).
func HTTPRedirectf(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values) {
	if !assert.HTTPRedirectf(t, handler, method, url, values) {
		t.FailNow()
	}
}

// HTTPSuccess asserts that a specified handler returns a success status code.
//
//  assert.HTTPSuccess(t, myHandler, "POST", "http://www.google.com", nil)
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPSuccess(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values) {
	if !assert.HTTPSuccess(t, handler, method, url, values) {
		t.FailNow()
	}
}

// HTTPSuccessf asserts that a specified handler returns a success status code.
//
//  assert.HTTPSuccessf(t, myHandler, "POST", "http://www.google.com", nil, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPSuccessf(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values) {
	if !assert.HTTPSuccessf(t, handler, method, url, values) {
		t.FailNow()
	}
}

// Implements asserts that an object is implemented by the specified interface.
//
//    assert.Implements(t, (*MyInterface)(nil), new(MyObject))
func Implements(t TestingT, interfaceObject interface{}, object interface{}, msgAndArgs ...interface{}) {
	if !assert.Implements(t, interfaceObject, object, msgAndArgs...) {
		t.FailNow()
	}
}

// Implementsf asserts that an object is implemented by the specified interface.
//
//    assert.Implementsf(t, (*MyInterface, "error message %s", "formatted")(nil), new(MyObject))
func Implementsf(t TestingT, interfaceObject interface{}, object interface{}, msg string, args ...interface{}) {
	if !assert.Implementsf(t, interfaceObject, object, msg, args...) {
		t.FailNow()
	}
}

// InDelta asserts that the two numerals are within delta of each other.
//
// 	 assert.InDelta(t, math.Pi, (22 / 7.0), 0.01)
//
// Returns whether the assertion was successful (true) or not (false).
func InDelta(t TestingT, expected interface{}, actual interface{}, delta float64, msgAndArgs ...interface{}) {
	if !assert.InDelta(t, expected, actual, delta, msgAndArgs...) {
		t.FailNow()
	}
}

// InDeltaSlice is the same as InDelta, except it compares two slices.
func InDeltaSlice(t TestingT, expected interface{}, actual interface{}, delta float64, msgAndArgs ...interface{}) {
	if !assert.InDeltaSlice(t, expected, actual, delta, msgAndArgs...) {
		t.FailNow()
	}
}

// InDeltaSlicef is the same as InDelta, except it compares two slices.
func InDeltaSlicef(t TestingT, expected interface{}, actual interface{}, delta float64, msg string, args ...interface{}) {
	if !assert.InDeltaSlicef(t, expected, actual, delta, msg, args...) {
		t.FailNow()
	}
}

// InDeltaf asserts that the two numerals are within delta of each other.
//
// 	 assert.InDeltaf(t, math.Pi, (22 / 7.0, "error message %s", "formatted"), 0.01)
//
// Returns whether the assertion was successful (true) or not (false).
func InDeltaf(t TestingT, expected interface{}, actual interface{}, delta float64, msg string, args ...interface{}) {
	if !assert.InDeltaf(t, expected, actual, delta, msg, args...) {
		t.FailNow()
	}
}

// InEpsilon asserts that expected and actual have a relative error less than epsilon
//
// Returns whether the assertion was successful (true) or not (false).
func InEpsilon(t TestingT, expected interface{}, actual interface{}, epsilon float64, msgAndArgs ...interface{}) {
	if !assert.InEpsilon(t, expected, actual, epsilon, msgAndArgs...) {
		t.FailNow()
	}
}

// InEpsilonSlice is the same as InEpsilon, except it compares each value from two slices.
func InEpsilonSlice(t TestingT, expected interface{}, actual interface{}, epsilon float64, msgAndArgs ...interface{}) {
	if !assert.InEpsilonSlice(t, expected, actual, epsilon, msgAndArgs...) {
		t.FailNow()
	}
}

// InEpsilonSlicef is the same as InEpsilon, except it compares each value from two slices.
func InEpsilonSlicef(t TestingT, expected interface{}, actual interface{}, epsilon float64, msg string, args ...interface{}) {
	if !assert.InEpsilonSlicef(t, expected, actual, epsilon, msg, args...) {
		t.FailNow()
	}
}

// InEpsilonf asserts that expected and actual have a relative error less than epsilon
//
// Returns whether the assertion was successful (true) or not (false).
func InEpsilonf(t TestingT, expected interface{}, actual interface{}, epsilon float64, msg string, args ...interface{}) {
	if !assert.InEpsilonf(t, expected, actual, epsilon, msg, args...) {
		t.FailNow()
	}
}

// IsType asserts that the specified objects are of the same type.
func IsType(t TestingT, expectedType interface{}, object interface{}, msgAndArgs ...interface{}) {
	if !assert.IsType(t, expectedType, object, msgAndArgs...) {
		t.FailNow()
	}
}

// IsTypef asserts that the specified objects are of the same type.
func IsTypef(t TestingT, expectedType interface{}, object interface{}, msg string, args ...interface{}) {
	if !assert.IsTypef(t, expectedType, object, msg, args...) {
		t.FailNow()
	}
}

// JSONEq asserts that two JSON strings are equivalent.
//
//  assert.JSONEq(t, `{"hello": "world", "foo": "bar"}`, `{"foo": "bar", "hello": "world"}`)
//
// Returns whether the assertion was successful (true) or not (false).
func JSONEq(t TestingT, expected string, actual string, msgAndArgs ...interface{}) {
	if !assert.JSONEq(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}

// JSONEqf asserts that two JSON strings are equivalent.
//
//  assert.JSONEqf(t, `{"hello": "world", "foo": "bar"}`, `{"foo": "bar", "hello": "world"}`, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func JSONEqf(t TestingT, expected string, actual string, msg string, args ...interface{}) {
	if !assert.JSONEqf(t, expected, actual, msg, args...) {
		t.FailNow()
	}
}

// Len asserts that the specified object has specific length.
// Len also fails if the object has a type that len() not accept.
//
//    assert.Len(t, mySlice, 3)
//
// Returns whether the assertion was successful (true) or not (false).
func Len(t TestingT, object interface{}, length int, msgAndArgs ...interface{}) {
	if !assert.Len(t, object, length, msgAndArgs...) {
		t.FailNow()
	}
}

// Lenf asserts that the specified object has specific length.
// Lenf also fails if the object has a type that len() not accept.
//
//    assert.Lenf(t, mySlice, 3, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Lenf(t TestingT, object interface{}, length int, msg string, args ...interface{}) {
	if !assert.Lenf(t, object, length, msg, args...) {
		t.FailNow()
	}
}

// Nil asserts that the specified object is nil.
//
//    assert.Nil(t, err)
//
// Returns whether the assertion was successful (true) or not (false).
func Nil(t TestingT, object interface{}, msgAndArgs ...interface{}) {
	if !assert.Nil(t, object, msgAndArgs...) {
		t.FailNow()
	}
}

// Nilf asserts that the specified object is nil.
//
//    assert.Nilf(t, err, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Nilf(t TestingT, object interface{}, msg string, args ...interface{}) {
	if !assert.Nilf(t, object, msg, args...) {
		t.FailNow()
	}
}

// NoError asserts that a function returned no error (i.e. `nil`).
//
//   actualObj, err := SomeFunction()
//   if assert.NoError(t, err) {
// 	   assert.Equal(t, expectedObj, actualObj)
//   }
//
// Returns whether the assertion was successful (true) or not (false).
func NoError(t TestingT, err error, msgAndArgs ...interface{}) {
	if !assert.NoError(t, err, msgAndArgs...) {
		t.FailNow()
	}
}

// NoErrorf asserts that a function returned no error (i.e. `nil`).
//
//   actualObj, err := SomeFunction()
//   if assert.NoErrorf(t, err, "error message %s", "formatted") {
// 	   assert.Equal(t, expectedObj, actualObj)
//   }
//
// Returns whether the assertion was successful (true) or not (false).
func NoErrorf(t TestingT, err error, msg string, args ...interface{}) {
	if !assert.NoErrorf(t, err, msg, args...) {
		t.FailNow()
	}
}

// NotContains asserts that the specified string, list(array, slice...) or map does NOT contain the
// specified substring or element.
//
//    assert.NotContains(t, "Hello World", "Earth")
//    assert.NotContains(t, ["Hello", "World"], "Earth")
//    assert.NotContains(t, {"Hello": "World"}, "Earth")
//
// Returns whether the assertion was successful (true) or not (false).
func NotContains(t TestingT, s interface{}, contains interface{}, msgAndArgs ...interface{}) {
	if !assert.NotContains(t, s, contains, msgAndArgs...) {
		t.FailNow()
	}
}

// NotContainsf asserts that the specified string, list(array, slice...) or map does NOT contain the
// specified substring or element.
//
//    assert.NotContainsf(t, "Hello World", "Earth", "error message %s", "formatted")
//    assert.NotContainsf(t, ["Hello", "World"], "Earth", "error message %s", "formatted")
//    assert.NotContainsf(t, {"Hello": "World"}, "Earth", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func NotContainsf(t TestingT, s interface{}, contains interface{}, msg string, args ...interface{}) {
	if !assert.NotContainsf(t, s, contains, msg, args...) {
		t.FailNow()
	}
}

// NotEmpty asserts that the specified object is NOT empty.  I.e. not nil, "", false, 0 or either
// a slice or a channel with len == 0.
//
//  if assert.NotEmpty(t, obj) {
//    assert.Equal(t, "two", obj[1])
//  }
//
// Returns whether the assertion was successful (true) or not (false).
func NotEmpty(t TestingT, object interface{}, msgAndArgs ...interface{}) {
	if !assert.NotEmpty(t, object, msgAndArgs...) {
		t.FailNow()
	}
}

// NotEmptyf asserts that the specified object is NOT empty.  I.e. not nil, "", false, 0 or either
// a slice or a channel with len == 0.
//
//  if assert.NotEmptyf(t, obj, "error message %s", "formatted") {
//    assert.Equal(t, "two", obj[1])
//  }
//
// Returns whether the assertion was successful (true) or not (false).
func NotEmptyf(t TestingT, object interface{}, msg string, args ...interface{}) {
	if !assert.NotEmptyf(t, object, msg, args...) {
		t.FailNow()
	}
}

// NotEqual asserts that the specified values are NOT equal.
//
//    assert.NotEqual(t, obj1, obj2)
//
// Returns whether the assertion was successful (true) or not (false).
//
// Pointer variable equality is determined based on the equality of the
// referenced values (as opposed to the memory addresses).
func NotEqual(t TestingT, expected interface{}, actual interface{}, msgAndArgs ...interface{}) {
	if !assert.NotEqual(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}

// NotEqualf asserts that the specified values are NOT equal.
//
//    assert.NotEqualf(t, obj1, obj2, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
//
// Pointer variable equality is determined based on the equality of the
// referenced values (as opposed to the memory addresses).
func NotEqualf(t TestingT, expected interface{}, actual interface{}, msg string, args ...interface{}) {
	if !assert.NotEqualf(t, expected, actual, msg, args...) {
		t.FailNow()
	}
}

// NotNil asserts that the specified object is not nil.
//
//    assert.NotNil(t, err)
//
// Returns whether the assertion was successful (true) or not (false).
func NotNil(t TestingT, object interface{}, msgAndArgs ...interface{}) {
	if !assert.NotNil(t, object, msgAndArgs...) {
		t.FailNow()
	}
}

// NotNilf asserts that the specified object is not nil.
//
//    assert.NotNilf(t, err, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func NotNilf(t TestingT, object interface{}, msg string, args ...interface{}) {
	if !assert.NotNilf(t, object, msg, args...) {
		t.FailNow()
	}
}

// NotPanics asserts that the code inside the specified PanicTestFunc does NOT panic.
//
//   assert.NotPanics(t, func(){ RemainCalm() })
//
// Returns whether the assertion was successful (true) or not (false).
func NotPanics(t TestingT, f assert.PanicTestFunc, msgAndArgs ...interface{}) {
	if !assert.NotPanics(t, f, msgAndArgs...) {
		t.FailNow()
	}
}

// NotPanicsf asserts that the code inside the specified PanicTestFunc does NOT panic.
//
//   assert.NotPanicsf(t, func(){ RemainCalm() }, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func NotPanicsf(t TestingT, f assert.PanicTestFunc, msg string, args ...interface{}) {
	if !assert.NotPanicsf(t, f, msg, args...) {
		t.FailNow()
	}
}

// NotRegexp asserts that a specified regexp does not match a string.
//
//  assert.NotRegexp(t, regexp.MustCompile("starts"), "it's starting")
//  assert.NotRegexp(t, "^start", "it's not starting")
//
// Returns whether the assertion was successful (true) or not (false).
func NotRegexp(t TestingT, rx interface{}, str interface{}, msgAndArgs ...interface{}) {
	if !assert.NotRegexp(t, rx, str, msgAndArgs...) {
		t.FailNow()
	}
}

// NotRegexpf asserts that a specified regexp does not match a string.
//
//  assert.NotRegexpf(t, regexp.MustCompile("starts", "error message %s", "formatted"), "it's starting")
//  assert.NotRegexpf(t, "^start", "it's not starting", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func NotRegexpf(t TestingT, rx interface{}, str interface{}, msg string, args ...interface{}) {
	if !assert.NotRegexpf(t, rx, str, msg, args...) {
		t.FailNow()
	}
}

// NotSubset asserts that the specified list(array, slice...) contains not all
// elements given in the specified subset(array, slice...).
//
//    assert.NotSubset(t, [1, 3, 4], [1, 2], "But [1, 3, 4] does not contain [1, 2]")
//
// Returns whether the assertion was successful (true) or not (false).
func NotSubset(t TestingT, list interface{}, subset interface{}, msgAndArgs ...interface{}) {
	if !assert.NotSubset(t, list, subset, msgAndArgs...) {
		t.FailNow()
	}
}

// NotSubsetf asserts that the specified list(array, slice...) contains not all
// elements given in the specified subset(array, slice...).
//
//    assert.NotSubsetf(t, [1, 3, 4], [1, 2], "But [1, 3, 4] does not contain [1, 2]", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func NotSubsetf(t TestingT, list interface{}, subset interface{}, msg string, args ...interface{}) {
	if !assert.NotSubsetf(t, list, subset, msg, args...) {
		t.FailNow()
	}
}

// NotZero asserts that i is not the zero value for its type and returns the truth.
func NotZero(t TestingT, i interface{}, msgAndArgs ...interface{}) {
	if !assert.NotZero(t, i, msgAndArgs...) {
		t.FailNow()
	}
}

// NotZerof asserts that i is not the zero value for its type and returns the truth.
func NotZerof(t TestingT, i interface{}, msg string, args ...interface{}) {
	if !assert.NotZerof(t, i, msg, args...) {
		t.FailNow()
	}
}

// Panics asserts that the code inside the specified PanicTestFunc panics.
//
//   assert.Panics(t, func(){ GoCrazy() })
//
// Returns whether the assertion was successful (true) or not (false).
func Panics(t TestingT, f assert.PanicTestFunc, msgAndArgs ...interface{}) {
	if !assert.Panics(t, f, msgAndArgs...) {
		t.FailNow()
	}
}

// PanicsWithValue asserts that the code inside the specified PanicTestFunc panics, and that
// the recovered panic value equals the expected panic value.
//
//   assert.PanicsWithValue(t, "crazy error", func(){ GoCrazy() })
//
// Returns whether the assertion was successful (true) or not (false).
func PanicsWithValue(t TestingT, expected interface{}, f assert.PanicTestFunc, msgAndArgs ...interface{}) {
	if !assert.PanicsWithValue(t, expected, f, msgAndArgs...) {
		t.FailNow()
	}
}

// PanicsWithValuef asserts that the code inside the specified PanicTestFunc panics, and that
// the recovered panic value equals the expected panic value.
//
//   assert.PanicsWithValuef(t, "crazy error", func(){ GoCrazy() }, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func PanicsWithValuef(t TestingT, expected interface{}, f assert.PanicTestFunc, msg string, args ...interface{}) {
	if !assert.PanicsWithValuef(t, expected, f, msg, args...) {
		t.FailNow()
	}
}

// Panicsf asserts that the code inside the specified PanicTestFunc panics.
//
//   assert.Panicsf(t, func(){ GoCrazy() }, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Panicsf(t TestingT, f assert.PanicTestFunc, msg string, args ...interface{}) {
	if !assert.Panicsf(t, f, msg, args...) {
		t.FailNow()
	}
}

// Regexp asserts that a specified regexp matches a string.
//
//  assert.Regexp(t, regexp.MustCompile("start"), "it's starting")
//  assert.Regexp(t, "start...$", "it's not starting")
//
// Returns whether the assertion was successful (true) or not (false).
func Regexp(t TestingT, rx interface{}, str interface{}, msgAndArgs ...interface{}) {
	if !assert.Regexp(t, rx, str, msgAndArgs...) {
		t.FailNow()
	}
}

// Regexpf asserts that a specified regexp matches a string.
//
//  assert.Regexpf(t, regexp.MustCompile("start", "error message %s", "formatted"), "it's starting")
//  assert.Regexpf(t, "start...$", "it's not starting", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Regexpf(t TestingT, rx interface{}, str interface{}, msg string, args ...interface{}) {
	if !assert.Regexpf(t, rx, str, msg, args...) {
		t.FailNow()
	}
}

// Subset asserts that the specified list(array, slice...) contains all
// elements given in the specified subset(array, slice...).
//
//    assert.Subset(t, [1, 2, 3], [1, 2], "But [1, 2, 3] does contain [1, 2]")
//
// Returns whether the assertion was successful (true) or not (false).
func Subset(t TestingT, list interface{}, subset interface{}, msgAndArgs ...interface{}) {
	if !assert.Subset(t, list, subset, msgAndArgs...) {
		t.FailNow()
	}
}

// Subsetf asserts that the specified list(array, slice...) contains all
// elements given in the specified subset(array, slice...).
//
//    assert.Subsetf(t, [1, 2, 3], [1, 2], "But [1, 2, 3] does contain [1, 2]", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Subsetf(t TestingT, list interface{}, subset interface{}, msg string, args ...interface{}) {
	if !assert.Subsetf(t, list, subset, msg, args...) {
		t.FailNow()
	}
}

// True asserts that the specified value is true.
//
//    assert.True(t, myBool)
//
// Returns whether the assertion was successful (true) or not (false).
func True(t TestingT, value bool, msgAndArgs ...interface{}) {
	if !assert.True(t, value, msgAndArgs...) {
		t.FailNow()
	}
}

// Truef asserts that the specified value is true.
//
//    assert.Truef(t, myBool, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Truef(t TestingT, value bool, msg string, args ...interface{}) {
	if !assert.Truef(t, value, msg, args...) {
		t.FailNow()
	}
}

// WithinDuration asserts that the two times are within duration delta of each other.
//
//   assert.WithinDuration(t, time.Now(), time.Now(), 10*time.Second)
//
// Returns whether the assertion was successful (true) or not (false).
func WithinDuration(t TestingT, expected time.Time, actual time.Time, delta time.Duration, msgAndArgs ...interface{}) {
	if !assert.WithinDuration(t, expected, actual, delta, msgAndArgs...) {
		t.FailNow()
	}
}

// WithinDurationf asserts that the two times are within duration delta of each other.
//
//   assert.WithinDurationf(t, time.Now(), time.Now(), 10*time.Second, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func WithinDurationf(t TestingT, expected time.Time, actual time.Time, delta time.Duration, msg string, args ...interface{}) {
	if !assert.WithinDurationf(t, expected, actual, delta, msg, args...) {
		t.FailNow()
	}
}

// Zero asserts that i is the zero value for its type and returns the truth.
func Zero(t TestingT, i interface{}, msgAndArgs ...interface{}) {
	if !assert.Zero(t, i, msgAndArgs...) {
		t.FailNow()
	}
}

// Zerof asserts that i is the zero value for its type and returns the truth.
func Zerof(t TestingT, i interface{}, msg string, args ...interface{}) {
	if !assert.Zerof(t, i, msg, args...) {
		t.FailNow()
	}
}
