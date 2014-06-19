// A set of comprehensive testing tools for use with the normal Go testing system.
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
//    assert.Equal(t, expected, actual [, message [, format-args])
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
//    assert.Error(t, errorObject [, message [, format-args]])
//
//    assert.NoError(t, errorObject [, message [, format-args]])
//
//    assert.Implements(t, (*MyInterface)(nil), new(MyObject) [,message [, format-args]])
//
//    assert.IsType(t, expectedObject, actualObject [, message [, format-args]])
//
//    assert.Contains(t, string, substring [, message [, format-args]])
//
//    assert.NotContains(t, string, substring [, message [, format-args]])
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

package assert
