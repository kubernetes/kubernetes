// Alternative testing tools which stop test execution if test failed.
//
// Example Usage
//
// The following is a complete example using require in a standard test function:
//    import (
//      "testing"
//      "github.com/stretchr/testify/require"
//    )
//
//    func TestSomething(t *testing.T) {
//
//      var a string = "Hello"
//      var b string = "Hello"
//
//      require.Equal(t, a, b, "The two words should be the same.")
//
//    }
//
// Assertions
//
// The `require` package have same global functions as in the `assert` package,
// but instead of returning a boolean result they call `t.FailNow()`.
//
// Every assertion function also takes an optional string message as the final argument,
// allowing custom error messages to be appended to the message the assertion method outputs.
//
// Here is an overview of the assert functions:
//
//    require.Equal(t, expected, actual [, message [, format-args])
//
//    require.NotEqual(t, notExpected, actual [, message [, format-args]])
//
//    require.True(t, actualBool [, message [, format-args]])
//
//    require.False(t, actualBool [, message [, format-args]])
//
//    require.Nil(t, actualObject [, message [, format-args]])
//
//    require.NotNil(t, actualObject [, message [, format-args]])
//
//    require.Empty(t, actualObject [, message [, format-args]])
//
//    require.NotEmpty(t, actualObject [, message [, format-args]])
//
//    require.Error(t, errorObject [, message [, format-args]])
//
//    require.NoError(t, errorObject [, message [, format-args]])
//
//    require.EqualError(t, theError, errString [, message [, format-args]])
//
//    require.Implements(t, (*MyInterface)(nil), new(MyObject) [,message [, format-args]])
//
//    require.IsType(t, expectedObject, actualObject [, message [, format-args]])
//
//    require.Contains(t, string, substring [, message [, format-args]])
//
//    require.NotContains(t, string, substring [, message [, format-args]])
//
//    require.Panics(t, func(){
//
//	    // call code that should panic
//
//    } [, message [, format-args]])
//
//    require.NotPanics(t, func(){
//
//	    // call code that should not panic
//
//    } [, message [, format-args]])
//
//    require.WithinDuration(t, timeA, timeB, deltaTime, [, message [, format-args]])
//
//    require.InDelta(t, numA, numB, delta, [, message [, format-args]])
//
//    require.InEpsilon(t, numA, numB, epsilon, [, message [, format-args]])
package require
