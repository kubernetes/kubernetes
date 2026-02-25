// Package assert provides a set of comprehensive testing tools for use with the normal Go testing system.
//
// # Note
//
// All functions in this package return a bool value indicating whether the assertion has passed.
//
// # Example Usage
//
// The following is a complete example using assert in a standard test function:
//
//	import (
//	  "testing"
//	  "github.com/stretchr/testify/assert"
//	)
//
//	func TestSomething(t *testing.T) {
//
//	  var a string = "Hello"
//	  var b string = "Hello"
//
//	  assert.Equal(t, a, b, "The two words should be the same.")
//
//	}
//
// if you assert many times, use the format below:
//
//	import (
//	  "testing"
//	  "github.com/stretchr/testify/assert"
//	)
//
//	func TestSomething(t *testing.T) {
//	  assert := assert.New(t)
//
//	  var a string = "Hello"
//	  var b string = "Hello"
//
//	  assert.Equal(a, b, "The two words should be the same.")
//	}
//
// # Assertions
//
// Assertions allow you to easily write test code, and are global funcs in the `assert` package.
// All assertion functions take, as the first argument, the `*testing.T` object provided by the
// testing framework. This allows the assertion funcs to write the failings and other details to
// the correct place.
//
// Every assertion function also takes an optional string message as the final argument,
// allowing custom error messages to be appended to the message the assertion method outputs.
package assert
