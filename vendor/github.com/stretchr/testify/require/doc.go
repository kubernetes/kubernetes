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
package require
