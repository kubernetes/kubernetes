/*
* CODE GENERATED AUTOMATICALLY WITH github.com/stretchr/testify/_codegen
* THIS FILE MUST NOT BE EDITED BY HAND
 */

package assert

import (
	http "net/http"
	url "net/url"
	time "time"
)

// Conditionf uses a Comparison to assert a complex condition.
func Conditionf(t TestingT, comp Comparison, msg string, args ...interface{}) bool {
	return Condition(t, comp, append([]interface{}{msg}, args...)...)
}

// Containsf asserts that the specified string, list(array, slice...) or map contains the
// specified substring or element.
//
//    assert.Containsf(t, "Hello World", "World", "error message %s", "formatted")
//    assert.Containsf(t, ["Hello", "World"], "World", "error message %s", "formatted")
//    assert.Containsf(t, {"Hello": "World"}, "Hello", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Containsf(t TestingT, s interface{}, contains interface{}, msg string, args ...interface{}) bool {
	return Contains(t, s, contains, append([]interface{}{msg}, args...)...)
}

// Emptyf asserts that the specified object is empty.  I.e. nil, "", false, 0 or either
// a slice or a channel with len == 0.
//
//  assert.Emptyf(t, obj, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Emptyf(t TestingT, object interface{}, msg string, args ...interface{}) bool {
	return Empty(t, object, append([]interface{}{msg}, args...)...)
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
func Equalf(t TestingT, expected interface{}, actual interface{}, msg string, args ...interface{}) bool {
	return Equal(t, expected, actual, append([]interface{}{msg}, args...)...)
}

// EqualErrorf asserts that a function returned an error (i.e. not `nil`)
// and that it is equal to the provided error.
//
//   actualObj, err := SomeFunction()
//   assert.EqualErrorf(t, err,  expectedErrorString, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func EqualErrorf(t TestingT, theError error, errString string, msg string, args ...interface{}) bool {
	return EqualError(t, theError, errString, append([]interface{}{msg}, args...)...)
}

// EqualValuesf asserts that two objects are equal or convertable to the same types
// and equal.
//
//    assert.EqualValuesf(t, uint32(123, "error message %s", "formatted"), int32(123))
//
// Returns whether the assertion was successful (true) or not (false).
func EqualValuesf(t TestingT, expected interface{}, actual interface{}, msg string, args ...interface{}) bool {
	return EqualValues(t, expected, actual, append([]interface{}{msg}, args...)...)
}

// Errorf asserts that a function returned an error (i.e. not `nil`).
//
//   actualObj, err := SomeFunction()
//   if assert.Errorf(t, err, "error message %s", "formatted") {
// 	   assert.Equal(t, expectedErrorf, err)
//   }
//
// Returns whether the assertion was successful (true) or not (false).
func Errorf(t TestingT, err error, msg string, args ...interface{}) bool {
	return Error(t, err, append([]interface{}{msg}, args...)...)
}

// Exactlyf asserts that two objects are equal is value and type.
//
//    assert.Exactlyf(t, int32(123, "error message %s", "formatted"), int64(123))
//
// Returns whether the assertion was successful (true) or not (false).
func Exactlyf(t TestingT, expected interface{}, actual interface{}, msg string, args ...interface{}) bool {
	return Exactly(t, expected, actual, append([]interface{}{msg}, args...)...)
}

// Failf reports a failure through
func Failf(t TestingT, failureMessage string, msg string, args ...interface{}) bool {
	return Fail(t, failureMessage, append([]interface{}{msg}, args...)...)
}

// FailNowf fails test
func FailNowf(t TestingT, failureMessage string, msg string, args ...interface{}) bool {
	return FailNow(t, failureMessage, append([]interface{}{msg}, args...)...)
}

// Falsef asserts that the specified value is false.
//
//    assert.Falsef(t, myBool, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Falsef(t TestingT, value bool, msg string, args ...interface{}) bool {
	return False(t, value, append([]interface{}{msg}, args...)...)
}

// HTTPBodyContainsf asserts that a specified handler returns a
// body that contains a string.
//
//  assert.HTTPBodyContainsf(t, myHandler, "www.google.com", nil, "I'm Feeling Lucky", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPBodyContainsf(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values, str interface{}) bool {
	return HTTPBodyContains(t, handler, method, url, values, str)
}

// HTTPBodyNotContainsf asserts that a specified handler returns a
// body that does not contain a string.
//
//  assert.HTTPBodyNotContainsf(t, myHandler, "www.google.com", nil, "I'm Feeling Lucky", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPBodyNotContainsf(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values, str interface{}) bool {
	return HTTPBodyNotContains(t, handler, method, url, values, str)
}

// HTTPErrorf asserts that a specified handler returns an error status code.
//
//  assert.HTTPErrorf(t, myHandler, "POST", "/a/b/c", url.Values{"a": []string{"b", "c"}}
//
// Returns whether the assertion was successful (true, "error message %s", "formatted") or not (false).
func HTTPErrorf(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values) bool {
	return HTTPError(t, handler, method, url, values)
}

// HTTPRedirectf asserts that a specified handler returns a redirect status code.
//
//  assert.HTTPRedirectf(t, myHandler, "GET", "/a/b/c", url.Values{"a": []string{"b", "c"}}
//
// Returns whether the assertion was successful (true, "error message %s", "formatted") or not (false).
func HTTPRedirectf(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values) bool {
	return HTTPRedirect(t, handler, method, url, values)
}

// HTTPSuccessf asserts that a specified handler returns a success status code.
//
//  assert.HTTPSuccessf(t, myHandler, "POST", "http://www.google.com", nil, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func HTTPSuccessf(t TestingT, handler http.HandlerFunc, method string, url string, values url.Values) bool {
	return HTTPSuccess(t, handler, method, url, values)
}

// Implementsf asserts that an object is implemented by the specified interface.
//
//    assert.Implementsf(t, (*MyInterface, "error message %s", "formatted")(nil), new(MyObject))
func Implementsf(t TestingT, interfaceObject interface{}, object interface{}, msg string, args ...interface{}) bool {
	return Implements(t, interfaceObject, object, append([]interface{}{msg}, args...)...)
}

// InDeltaf asserts that the two numerals are within delta of each other.
//
// 	 assert.InDeltaf(t, math.Pi, (22 / 7.0, "error message %s", "formatted"), 0.01)
//
// Returns whether the assertion was successful (true) or not (false).
func InDeltaf(t TestingT, expected interface{}, actual interface{}, delta float64, msg string, args ...interface{}) bool {
	return InDelta(t, expected, actual, delta, append([]interface{}{msg}, args...)...)
}

// InDeltaSlicef is the same as InDelta, except it compares two slices.
func InDeltaSlicef(t TestingT, expected interface{}, actual interface{}, delta float64, msg string, args ...interface{}) bool {
	return InDeltaSlice(t, expected, actual, delta, append([]interface{}{msg}, args...)...)
}

// InEpsilonf asserts that expected and actual have a relative error less than epsilon
//
// Returns whether the assertion was successful (true) or not (false).
func InEpsilonf(t TestingT, expected interface{}, actual interface{}, epsilon float64, msg string, args ...interface{}) bool {
	return InEpsilon(t, expected, actual, epsilon, append([]interface{}{msg}, args...)...)
}

// InEpsilonSlicef is the same as InEpsilon, except it compares each value from two slices.
func InEpsilonSlicef(t TestingT, expected interface{}, actual interface{}, epsilon float64, msg string, args ...interface{}) bool {
	return InEpsilonSlice(t, expected, actual, epsilon, append([]interface{}{msg}, args...)...)
}

// IsTypef asserts that the specified objects are of the same type.
func IsTypef(t TestingT, expectedType interface{}, object interface{}, msg string, args ...interface{}) bool {
	return IsType(t, expectedType, object, append([]interface{}{msg}, args...)...)
}

// JSONEqf asserts that two JSON strings are equivalent.
//
//  assert.JSONEqf(t, `{"hello": "world", "foo": "bar"}`, `{"foo": "bar", "hello": "world"}`, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func JSONEqf(t TestingT, expected string, actual string, msg string, args ...interface{}) bool {
	return JSONEq(t, expected, actual, append([]interface{}{msg}, args...)...)
}

// Lenf asserts that the specified object has specific length.
// Lenf also fails if the object has a type that len() not accept.
//
//    assert.Lenf(t, mySlice, 3, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Lenf(t TestingT, object interface{}, length int, msg string, args ...interface{}) bool {
	return Len(t, object, length, append([]interface{}{msg}, args...)...)
}

// Nilf asserts that the specified object is nil.
//
//    assert.Nilf(t, err, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Nilf(t TestingT, object interface{}, msg string, args ...interface{}) bool {
	return Nil(t, object, append([]interface{}{msg}, args...)...)
}

// NoErrorf asserts that a function returned no error (i.e. `nil`).
//
//   actualObj, err := SomeFunction()
//   if assert.NoErrorf(t, err, "error message %s", "formatted") {
// 	   assert.Equal(t, expectedObj, actualObj)
//   }
//
// Returns whether the assertion was successful (true) or not (false).
func NoErrorf(t TestingT, err error, msg string, args ...interface{}) bool {
	return NoError(t, err, append([]interface{}{msg}, args...)...)
}

// NotContainsf asserts that the specified string, list(array, slice...) or map does NOT contain the
// specified substring or element.
//
//    assert.NotContainsf(t, "Hello World", "Earth", "error message %s", "formatted")
//    assert.NotContainsf(t, ["Hello", "World"], "Earth", "error message %s", "formatted")
//    assert.NotContainsf(t, {"Hello": "World"}, "Earth", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func NotContainsf(t TestingT, s interface{}, contains interface{}, msg string, args ...interface{}) bool {
	return NotContains(t, s, contains, append([]interface{}{msg}, args...)...)
}

// NotEmptyf asserts that the specified object is NOT empty.  I.e. not nil, "", false, 0 or either
// a slice or a channel with len == 0.
//
//  if assert.NotEmptyf(t, obj, "error message %s", "formatted") {
//    assert.Equal(t, "two", obj[1])
//  }
//
// Returns whether the assertion was successful (true) or not (false).
func NotEmptyf(t TestingT, object interface{}, msg string, args ...interface{}) bool {
	return NotEmpty(t, object, append([]interface{}{msg}, args...)...)
}

// NotEqualf asserts that the specified values are NOT equal.
//
//    assert.NotEqualf(t, obj1, obj2, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
//
// Pointer variable equality is determined based on the equality of the
// referenced values (as opposed to the memory addresses).
func NotEqualf(t TestingT, expected interface{}, actual interface{}, msg string, args ...interface{}) bool {
	return NotEqual(t, expected, actual, append([]interface{}{msg}, args...)...)
}

// NotNilf asserts that the specified object is not nil.
//
//    assert.NotNilf(t, err, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func NotNilf(t TestingT, object interface{}, msg string, args ...interface{}) bool {
	return NotNil(t, object, append([]interface{}{msg}, args...)...)
}

// NotPanicsf asserts that the code inside the specified PanicTestFunc does NOT panic.
//
//   assert.NotPanicsf(t, func(){ RemainCalm() }, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func NotPanicsf(t TestingT, f PanicTestFunc, msg string, args ...interface{}) bool {
	return NotPanics(t, f, append([]interface{}{msg}, args...)...)
}

// NotRegexpf asserts that a specified regexp does not match a string.
//
//  assert.NotRegexpf(t, regexp.MustCompile("starts", "error message %s", "formatted"), "it's starting")
//  assert.NotRegexpf(t, "^start", "it's not starting", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func NotRegexpf(t TestingT, rx interface{}, str interface{}, msg string, args ...interface{}) bool {
	return NotRegexp(t, rx, str, append([]interface{}{msg}, args...)...)
}

// NotSubsetf asserts that the specified list(array, slice...) contains not all
// elements given in the specified subset(array, slice...).
//
//    assert.NotSubsetf(t, [1, 3, 4], [1, 2], "But [1, 3, 4] does not contain [1, 2]", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func NotSubsetf(t TestingT, list interface{}, subset interface{}, msg string, args ...interface{}) bool {
	return NotSubset(t, list, subset, append([]interface{}{msg}, args...)...)
}

// NotZerof asserts that i is not the zero value for its type and returns the truth.
func NotZerof(t TestingT, i interface{}, msg string, args ...interface{}) bool {
	return NotZero(t, i, append([]interface{}{msg}, args...)...)
}

// Panicsf asserts that the code inside the specified PanicTestFunc panics.
//
//   assert.Panicsf(t, func(){ GoCrazy() }, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Panicsf(t TestingT, f PanicTestFunc, msg string, args ...interface{}) bool {
	return Panics(t, f, append([]interface{}{msg}, args...)...)
}

// PanicsWithValuef asserts that the code inside the specified PanicTestFunc panics, and that
// the recovered panic value equals the expected panic value.
//
//   assert.PanicsWithValuef(t, "crazy error", func(){ GoCrazy() }, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func PanicsWithValuef(t TestingT, expected interface{}, f PanicTestFunc, msg string, args ...interface{}) bool {
	return PanicsWithValue(t, expected, f, append([]interface{}{msg}, args...)...)
}

// Regexpf asserts that a specified regexp matches a string.
//
//  assert.Regexpf(t, regexp.MustCompile("start", "error message %s", "formatted"), "it's starting")
//  assert.Regexpf(t, "start...$", "it's not starting", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Regexpf(t TestingT, rx interface{}, str interface{}, msg string, args ...interface{}) bool {
	return Regexp(t, rx, str, append([]interface{}{msg}, args...)...)
}

// Subsetf asserts that the specified list(array, slice...) contains all
// elements given in the specified subset(array, slice...).
//
//    assert.Subsetf(t, [1, 2, 3], [1, 2], "But [1, 2, 3] does contain [1, 2]", "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Subsetf(t TestingT, list interface{}, subset interface{}, msg string, args ...interface{}) bool {
	return Subset(t, list, subset, append([]interface{}{msg}, args...)...)
}

// Truef asserts that the specified value is true.
//
//    assert.Truef(t, myBool, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func Truef(t TestingT, value bool, msg string, args ...interface{}) bool {
	return True(t, value, append([]interface{}{msg}, args...)...)
}

// WithinDurationf asserts that the two times are within duration delta of each other.
//
//   assert.WithinDurationf(t, time.Now(), time.Now(), 10*time.Second, "error message %s", "formatted")
//
// Returns whether the assertion was successful (true) or not (false).
func WithinDurationf(t TestingT, expected time.Time, actual time.Time, delta time.Duration, msg string, args ...interface{}) bool {
	return WithinDuration(t, expected, actual, delta, append([]interface{}{msg}, args...)...)
}

// Zerof asserts that i is the zero value for its type and returns the truth.
func Zerof(t TestingT, i interface{}, msg string, args ...interface{}) bool {
	return Zero(t, i, append([]interface{}{msg}, args...)...)
}
