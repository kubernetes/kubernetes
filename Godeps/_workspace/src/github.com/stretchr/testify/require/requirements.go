package require

import (
	"time"

	"github.com/stretchr/testify/assert"
)

type TestingT interface {
	Errorf(format string, args ...interface{})
	FailNow()
}

// Fail reports a failure through
func FailNow(t TestingT, failureMessage string, msgAndArgs ...interface{}) {
	assert.Fail(t, failureMessage, msgAndArgs...)
	t.FailNow()
}

// Implements asserts that an object is implemented by the specified interface.
//
//    require.Implements(t, (*MyInterface)(nil), new(MyObject), "MyObject")
func Implements(t TestingT, interfaceObject interface{}, object interface{}, msgAndArgs ...interface{}) {
	if !assert.Implements(t, interfaceObject, object, msgAndArgs...) {
		t.FailNow()
	}
}

// IsType asserts that the specified objects are of the same type.
func IsType(t TestingT, expectedType interface{}, object interface{}, msgAndArgs ...interface{}) {
	if !assert.IsType(t, expectedType, object, msgAndArgs...) {
		t.FailNow()
	}
}

// Equal asserts that two objects are equal.
//
//    require.Equal(t, 123, 123, "123 and 123 should be equal")
func Equal(t TestingT, expected, actual interface{}, msgAndArgs ...interface{}) {
	if !assert.Equal(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}

// EqualValues asserts that two objects are equal or convertable to each other.
//
//    require.EqualValues(t, uint32(123), int32(123), "123 and 123 should be equal")
func EqualValues(t TestingT, expected, actual interface{}, msgAndArgs ...interface{}) {
	if !assert.EqualValues(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}

// Exactly asserts that two objects are equal is value and type.
//
//    require.Exactly(t, int32(123), int64(123), "123 and 123 should NOT be equal")
func Exactly(t TestingT, expected, actual interface{}, msgAndArgs ...interface{}) {
	if !assert.Exactly(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}

// NotNil asserts that the specified object is not nil.
//
//    require.NotNil(t, err, "err should be something")
func NotNil(t TestingT, object interface{}, msgAndArgs ...interface{}) {
	if !assert.NotNil(t, object, msgAndArgs...) {
		t.FailNow()
	}
}

// Nil asserts that the specified object is nil.
//
//    require.Nil(t, err, "err should be nothing")
func Nil(t TestingT, object interface{}, msgAndArgs ...interface{}) {
	if !assert.Nil(t, object, msgAndArgs...) {
		t.FailNow()
	}
}

// Empty asserts that the specified object is empty.  I.e. nil, "", false, 0 or either
// a slice or a channel with len == 0.
//
// require.Empty(t, obj)
func Empty(t TestingT, object interface{}, msgAndArgs ...interface{}) {
	if !assert.Empty(t, object, msgAndArgs...) {
		t.FailNow()
	}
}

// NotEmpty asserts that the specified object is NOT empty.  I.e. not nil, "", false, 0 or either
// a slice or a channel with len == 0.
//
// require.NotEmpty(t, obj)
// require.Equal(t, "one", obj[0])
func NotEmpty(t TestingT, object interface{}, msgAndArgs ...interface{}) {
	if !assert.NotEmpty(t, object, msgAndArgs...) {
		t.FailNow()
	}
}

// Len asserts that the specified object has specific length.
// Len also fails if the object has a type that len() not accept.
//
//    require.Len(t, mySlice, 3, "The size of slice is not 3")
func Len(t TestingT, object interface{}, length int, msgAndArgs ...interface{}) {
	if !assert.Len(t, object, length, msgAndArgs...) {
		t.FailNow()
	}
}

// True asserts that the specified value is true.
//
//    require.True(t, myBool, "myBool should be true")
func True(t TestingT, value bool, msgAndArgs ...interface{}) {
	if !assert.True(t, value, msgAndArgs...) {
		t.FailNow()
	}
}

// False asserts that the specified value is true.
//
//    require.False(t, myBool, "myBool should be false")
func False(t TestingT, value bool, msgAndArgs ...interface{}) {
	if !assert.False(t, value, msgAndArgs...) {
		t.FailNow()
	}
}

// NotEqual asserts that the specified values are NOT equal.
//
//    require.NotEqual(t, obj1, obj2, "two objects shouldn't be equal")
func NotEqual(t TestingT, expected, actual interface{}, msgAndArgs ...interface{}) {
	if !assert.NotEqual(t, expected, actual, msgAndArgs...) {
		t.FailNow()
	}
}

// Contains asserts that the specified string contains the specified substring.
//
//    require.Contains(t, "Hello World", "World", "But 'Hello World' does contain 'World'")
func Contains(t TestingT, s, contains interface{}, msgAndArgs ...interface{}) {
	if !assert.Contains(t, s, contains, msgAndArgs...) {
		t.FailNow()
	}
}

// NotContains asserts that the specified string does NOT contain the specified substring.
//
//    require.NotContains(t, "Hello World", "Earth", "But 'Hello World' does NOT contain 'Earth'")
func NotContains(t TestingT, s, contains interface{}, msgAndArgs ...interface{}) {
	if !assert.NotContains(t, s, contains, msgAndArgs...) {
		t.FailNow()
	}
}

// Condition uses a Comparison to assert a complex condition.
func Condition(t TestingT, comp assert.Comparison, msgAndArgs ...interface{}) {
	if !assert.Condition(t, comp, msgAndArgs...) {
		t.FailNow()
	}
}

// Panics asserts that the code inside the specified PanicTestFunc panics.
//
//   require.Panics(t, func(){
//     GoCrazy()
//   }, "Calling GoCrazy() should panic")
func Panics(t TestingT, f assert.PanicTestFunc, msgAndArgs ...interface{}) {
	if !assert.Panics(t, f, msgAndArgs...) {
		t.FailNow()
	}
}

// NotPanics asserts that the code inside the specified PanicTestFunc does NOT panic.
//
//   require.NotPanics(t, func(){
//     RemainCalm()
//   }, "Calling RemainCalm() should NOT panic")
func NotPanics(t TestingT, f assert.PanicTestFunc, msgAndArgs ...interface{}) {
	if !assert.NotPanics(t, f, msgAndArgs...) {
		t.FailNow()
	}
}

// WithinDuration asserts that the two times are within duration delta of each other.
//
//   require.WithinDuration(t, time.Now(), time.Now(), 10*time.Second, "The difference should not be more than 10s")
func WithinDuration(t TestingT, expected, actual time.Time, delta time.Duration, msgAndArgs ...interface{}) {
	if !assert.WithinDuration(t, expected, actual, delta, msgAndArgs...) {
		t.FailNow()
	}
}

// InDelta asserts that the two numerals are within delta of each other.
//
//   require.InDelta(t, math.Pi, (22 / 7.0), 0.01)
func InDelta(t TestingT, expected, actual interface{}, delta float64, msgAndArgs ...interface{}) {
	if !assert.InDelta(t, expected, actual, delta, msgAndArgs...) {
		t.FailNow()
	}
}

// InEpsilon asserts that expected and actual have a relative error less than epsilon
func InEpsilon(t TestingT, expected, actual interface{}, epsilon float64, msgAndArgs ...interface{}) {
	if !assert.InEpsilon(t, expected, actual, epsilon, msgAndArgs...) {
		t.FailNow()
	}
}

// Regexp asserts that a specified regexp matches a string.
//
//  require.Regexp(t, regexp.MustCompile("start"), "it's starting")
//  require.Regexp(t, "start...$", "it's not starting")
func Regexp(t TestingT, rx interface{}, str interface{}, msgAndArgs ...interface{}) {
	if !assert.Regexp(t, rx, str, msgAndArgs...) {
		t.FailNow()
	}
}

// NotRegexp asserts that a specified regexp does not match a string.
//
//  require.NotRegexp(t, regexp.MustCompile("starts"), "it's starting")
//  require.NotRegexp(t, "^start", "it's not starting")
func NotRegexp(t TestingT, rx interface{}, str interface{}, msgAndArgs ...interface{}) {
	if !assert.NotRegexp(t, rx, str, msgAndArgs...) {
		t.FailNow()
	}
}

/*
	Errors
*/

// NoError asserts that a function returned no error (i.e. `nil`).
//
//   actualObj, err := SomeFunction()
//   require.NoError(t, err)
//   require.Equal(t, actualObj, expectedObj)
//
// Returns whether the assertion was successful (true) or not (false).
func NoError(t TestingT, err error, msgAndArgs ...interface{}) {
	if !assert.NoError(t, err, msgAndArgs...) {
		t.FailNow()
	}
}

// Error asserts that a function returned an error (i.e. not `nil`).
//
//   actualObj, err := SomeFunction()
//   require.Error(t, err, "An error was expected")
//   require.Equal(t, err, expectedError)
//   }
func Error(t TestingT, err error, msgAndArgs ...interface{}) {
	if !assert.Error(t, err, msgAndArgs...) {
		t.FailNow()
	}
}

// EqualError asserts that a function returned an error (i.e. not `nil`)
// and that it is equal to the provided error.
//
//   actualObj, err := SomeFunction()
//   require.Error(t, err, "An error was expected")
//   require.Equal(t, err, expectedError)
//   }
func EqualError(t TestingT, theError error, errString string, msgAndArgs ...interface{}) {
	if !assert.EqualError(t, theError, errString, msgAndArgs...) {
		t.FailNow()
	}
}
