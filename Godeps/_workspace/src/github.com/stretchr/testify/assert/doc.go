// Package assert provides a set of comprehensive testing tools for use with the normal Go testing system.
//
// Example Usage
//
// The following is a complete example using assert in a standard test function:
//    import (
//      "testing"
//      "github.com/stretchr/testify/assert"
//    )
//
//    func TestSomething(t *testing.T) {
//
//      var a string = "Hello"
//      var b string = "Hello"
//
//      assert.Equal(t, a, b, "The two words should be the same.")
//
//    }
//
// if you assert many times, use the below:
//
//    import (
//      "testing"
//      "github.com/stretchr/testify/assert"
//    )
//
//    func TestSomething(t *testing.T) {
//      assert := assert.New(t)
//
//      var a string = "Hello"
//      var b string = "Hello"
//
//      assert.Equal(a, b, "The two words should be the same.")
//    }
//
// Assertions
//
// Assertions allow you to easily write test code, and are global funcs in the `assert` package.
// All assertion functions take, as the first argument, the `*testing.T` object provided by the
// testing framework. This allows the assertion funcs to write the failings and other details to
// the correct place.
//
// Every assertion function also takes an optional string message as the final argument,
// allowing custom error messages to be appended to the message the assertion method outputs.
//
// Here is an overview of the assert functions:
//
//    assert.Equal(t, expected, actual [, message [, format-args]])
//
//    assert.EqualValues(t, expected, actual [, message [, format-args]])
//
//    assert.NotEqual(t, notExpected, actual [, message [, format-args]])
//
//    assert.True(t, actualBool [, message [, format-args]])
//
//    assert.False(t, actualBool [, message [, format-args]])
//
//    assert.Nil(t, actualObject [, message [, format-args]])
//
//    assert.NotNil(t, actualObject [, message [, format-args]])
//
//    assert.Empty(t, actualObject [, message [, format-args]])
//
//    assert.NotEmpty(t, actualObject [, message [, format-args]])
//
//    assert.Len(t, actualObject, expectedLength, [, message [, format-args]])
//
//    assert.Error(t, errorObject [, message [, format-args]])
//
//    assert.NoError(t, errorObject [, message [, format-args]])
//
//    assert.EqualError(t, theError, errString [, message [, format-args]])
//
//    assert.Implements(t, (*MyInterface)(nil), new(MyObject) [,message [, format-args]])
//
//    assert.IsType(t, expectedObject, actualObject [, message [, format-args]])
//
//    assert.Contains(t, stringOrSlice, substringOrElement [, message [, format-args]])
//
//    assert.NotContains(t, stringOrSlice, substringOrElement [, message [, format-args]])
//
//    assert.Panics(t, func(){
//
//	    // call code that should panic
//
//    } [, message [, format-args]])
//
//    assert.NotPanics(t, func(){
//
//	    // call code that should not panic
//
//    } [, message [, format-args]])
//
//    assert.WithinDuration(t, timeA, timeB, deltaTime, [, message [, format-args]])
//
//    assert.InDelta(t, numA, numB, delta, [, message [, format-args]])
//
//    assert.InEpsilon(t, numA, numB, epsilon, [, message [, format-args]])
//
// assert package contains Assertions object. it has assertion methods.
//
// Here is an overview of the assert functions:
//    assert.Equal(expected, actual [, message [, format-args]])
//
//    assert.EqualValues(expected, actual [, message [, format-args]])
//
//    assert.NotEqual(notExpected, actual [, message [, format-args]])
//
//    assert.True(actualBool [, message [, format-args]])
//
//    assert.False(actualBool [, message [, format-args]])
//
//    assert.Nil(actualObject [, message [, format-args]])
//
//    assert.NotNil(actualObject [, message [, format-args]])
//
//    assert.Empty(actualObject [, message [, format-args]])
//
//    assert.NotEmpty(actualObject [, message [, format-args]])
//
//    assert.Len(actualObject, expectedLength, [, message [, format-args]])
//
//    assert.Error(errorObject [, message [, format-args]])
//
//    assert.NoError(errorObject [, message [, format-args]])
//
//    assert.EqualError(theError, errString [, message [, format-args]])
//
//    assert.Implements((*MyInterface)(nil), new(MyObject) [,message [, format-args]])
//
//    assert.IsType(expectedObject, actualObject [, message [, format-args]])
//
//    assert.Contains(stringOrSlice, substringOrElement [, message [, format-args]])
//
//    assert.NotContains(stringOrSlice, substringOrElement [, message [, format-args]])
//
//    assert.Panics(func(){
//
//	    // call code that should panic
//
//    } [, message [, format-args]])
//
//    assert.NotPanics(func(){
//
//	    // call code that should not panic
//
//    } [, message [, format-args]])
//
//    assert.WithinDuration(timeA, timeB, deltaTime, [, message [, format-args]])
//
//    assert.InDelta(numA, numB, delta, [, message [, format-args]])
//
//    assert.InEpsilon(numA, numB, epsilon, [, message [, format-args]])
package assert
