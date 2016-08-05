package require

import (
	"time"

	"github.com/stretchr/testify/assert"
)

type Assertions struct {
	t TestingT
}

func New(t TestingT) *Assertions {
	return &Assertions{
		t: t,
	}
}

// Fail reports a failure through
func (a *Assertions) Fail(failureMessage string, msgAndArgs ...interface{}) {
	FailNow(a.t, failureMessage, msgAndArgs...)
}

// Implements asserts that an object is implemented by the specified interface.

func (a *Assertions) Implements(interfaceObject interface{}, object interface{}, msgAndArgs ...interface{}) {
	Implements(a.t, interfaceObject, object, msgAndArgs...)
}

// IsType asserts that the specified objects are of the same type.
func (a *Assertions) IsType(expectedType interface{}, object interface{}, msgAndArgs ...interface{}) {
	IsType(a.t, expectedType, object, msgAndArgs...)
}

// Equal asserts that two objects are equal.
//
//    require.Equal(123, 123, "123 and 123 should be equal")
func (a *Assertions) Equal(expected, actual interface{}, msgAndArgs ...interface{}) {
	Equal(a.t, expected, actual, msgAndArgs...)
}

// Exactly asserts that two objects are equal is value and type.
//
//    require.Exactly(int32(123), int64(123), "123 and 123 should NOT be equal")
func (a *Assertions) Exactly(expected, actual interface{}, msgAndArgs ...interface{}) {
	Exactly(a.t, expected, actual, msgAndArgs...)
}

// NotNil asserts that the specified object is not nil.
//
//    require.NotNil(err, "err should be something")
func (a *Assertions) NotNil(object interface{}, msgAndArgs ...interface{}) {
	NotNil(a.t, object, msgAndArgs...)
}

// Nil asserts that the specified object is nil.
//
//    require.Nil(err, "err should be nothing")
func (a *Assertions) Nil(object interface{}, msgAndArgs ...interface{}) {
	Nil(a.t, object, msgAndArgs...)
}

// Empty asserts that the specified object is empty.  I.e. nil, "", false, 0 or a
// slice with len == 0.
//
// require.Empty(obj)
func (a *Assertions) Empty(object interface{}, msgAndArgs ...interface{}) {
	Empty(a.t, object, msgAndArgs...)
}

// Empty asserts that the specified object is NOT empty.  I.e. not nil, "", false, 0 or a
// slice with len == 0.
//
// if require.NotEmpty(obj) {
//   require.Equal("two", obj[1])
// }
func (a *Assertions) NotEmpty(object interface{}, msgAndArgs ...interface{}) {
	NotEmpty(a.t, object, msgAndArgs...)
}

// Len asserts that the specified object has specific length.
// Len also fails if the object has a type that len() not accept.
//
//    require.Len(mySlice, 3, "The size of slice is not 3")
func (a *Assertions) Len(object interface{}, length int, msgAndArgs ...interface{}) {
	Len(a.t, object, length, msgAndArgs...)
}

// True asserts that the specified value is true.
//
//    require.True(myBool, "myBool should be true")
func (a *Assertions) True(value bool, msgAndArgs ...interface{}) {
	True(a.t, value, msgAndArgs...)
}

// False asserts that the specified value is false.
//
//    require.False(myBool, "myBool should be false")
func (a *Assertions) False(value bool, msgAndArgs ...interface{}) {
	False(a.t, value, msgAndArgs...)
}

// NotEqual asserts that the specified values are NOT equal.
//
//    require.NotEqual(obj1, obj2, "two objects shouldn't be equal")
func (a *Assertions) NotEqual(expected, actual interface{}, msgAndArgs ...interface{}) {
	NotEqual(a.t, expected, actual, msgAndArgs...)
}

// Contains asserts that the specified string contains the specified substring.
//
//    require.Contains("Hello World", "World", "But 'Hello World' does contain 'World'")
func (a *Assertions) Contains(s, contains interface{}, msgAndArgs ...interface{}) {
	Contains(a.t, s, contains, msgAndArgs...)
}

// NotContains asserts that the specified string does NOT contain the specified substring.
//
//    require.NotContains("Hello World", "Earth", "But 'Hello World' does NOT contain 'Earth'")
func (a *Assertions) NotContains(s, contains interface{}, msgAndArgs ...interface{}) {
	NotContains(a.t, s, contains, msgAndArgs...)
}

// Uses a Comparison to assert a complex condition.
func (a *Assertions) Condition(comp assert.Comparison, msgAndArgs ...interface{}) {
	Condition(a.t, comp, msgAndArgs...)
}

// Panics asserts that the code inside the specified PanicTestFunc panics.
//
//   require.Panics(func(){
//     GoCrazy()
//   }, "Calling GoCrazy() should panic")
func (a *Assertions) Panics(f assert.PanicTestFunc, msgAndArgs ...interface{}) {
	Panics(a.t, f, msgAndArgs...)
}

// NotPanics asserts that the code inside the specified PanicTestFunc does NOT panic.
//
//   require.NotPanics(func(){
//     RemainCalm()
//   }, "Calling RemainCalm() should NOT panic")
func (a *Assertions) NotPanics(f assert.PanicTestFunc, msgAndArgs ...interface{}) {
	NotPanics(a.t, f, msgAndArgs...)
}

// WithinDuration asserts that the two times are within duration delta of each other.
//
//   require.WithinDuration(time.Now(), time.Now(), 10*time.Second, "The difference should not be more than 10s")
func (a *Assertions) WithinDuration(expected, actual time.Time, delta time.Duration, msgAndArgs ...interface{}) {
	WithinDuration(a.t, expected, actual, delta, msgAndArgs...)
}

// InDelta asserts that the two numerals are within delta of each other.
//
// 	 require.InDelta(t, math.Pi, (22 / 7.0), 0.01)
func (a *Assertions) InDelta(expected, actual interface{}, delta float64, msgAndArgs ...interface{}) {
	InDelta(a.t, expected, actual, delta, msgAndArgs...)
}

// InEpsilon asserts that expected and actual have a relative error less than epsilon
func (a *Assertions) InEpsilon(expected, actual interface{}, epsilon float64, msgAndArgs ...interface{}) {
	InEpsilon(a.t, expected, actual, epsilon, msgAndArgs...)
}

// NoError asserts that a function returned no error (i.e. `nil`).
//
//   actualObj, err := SomeFunction()
//   if require.NoError(err) {
//	   require.Equal(actualObj, expectedObj)
//   }
func (a *Assertions) NoError(theError error, msgAndArgs ...interface{}) {
	NoError(a.t, theError, msgAndArgs...)
}

// Error asserts that a function returned an error (i.e. not `nil`).
//
//   actualObj, err := SomeFunction()
//   if require.Error(err, "An error was expected") {
//	   require.Equal(err, expectedError)
//   }
func (a *Assertions) Error(theError error, msgAndArgs ...interface{}) {
	Error(a.t, theError, msgAndArgs...)
}

// EqualError asserts that a function returned an error (i.e. not `nil`)
// and that it is equal to the provided error.
//
//   actualObj, err := SomeFunction()
//   if require.Error(err, "An error was expected") {
//	   require.Equal(err, expectedError)
//   }
func (a *Assertions) EqualError(theError error, errString string, msgAndArgs ...interface{}) {
	EqualError(a.t, theError, errString, msgAndArgs...)
}

// Regexp asserts that a specified regexp matches a string.
//
//  require.Regexp(t, regexp.MustCompile("start"), "it's starting")
//  require.Regexp(t, "start...$", "it's not starting")
func (a *Assertions) Regexp(rx interface{}, str interface{}, msgAndArgs ...interface{}) {
	Regexp(a.t, rx, str, msgAndArgs...)
}

// NotRegexp asserts that a specified regexp does not match a string.
//
//  require.NotRegexp(t, regexp.MustCompile("starts"), "it's starting")
//  require.NotRegexp(t, "^start", "it's not starting")
func (a *Assertions) NotRegexp(rx interface{}, str interface{}, msgAndArgs ...interface{}) {
	NotRegexp(a.t, rx, str, msgAndArgs...)
}

// Zero asserts that i is the zero value for its type and returns the truth.
func (a *Assertions) Zero(i interface{}, msgAndArgs ...interface{}) {
	Zero(a.t, i, msgAndArgs...)
}

// NotZero asserts that i is not the zero value for its type and returns the truth.
func (a *Assertions) NotZero(i interface{}, msgAndArgs ...interface{}) {
	NotZero(a.t, i, msgAndArgs...)
}

// JSONEq asserts that two JSON strings are equivalent.
//
//  assert.JSONEq(t, `{"hello": "world", "foo": "bar"}`, `{"foo": "bar", "hello": "world"}`)
//
// Returns whether the assertion was successful (true) or not (false).
func (a *Assertions) JSONEq(expected string, actual string, msgAndArgs ...interface{}) {
	JSONEq(a.t, expected, actual, msgAndArgs...)
}
