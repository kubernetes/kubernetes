// The suite package contains logic for creating testing suite structs
// and running the methods on those structs as tests.  The most useful
// piece of this package is that you can create setup/teardown methods
// on your testing suites, which will run before/after the whole suite
// or individual tests (depending on which interface(s) you
// implement).
//
// A testing suite is usually built by first extending the built-in
// suite functionality from suite.Suite in testify.  Alternatively,
// you could reproduce that logic on your own if you wanted (you
// just need to implement the TestingSuite interface from
// suite/interfaces.go).
//
// After that, you can implement any of the interfaces in
// suite/interfaces.go to add setup/teardown functionality to your
// suite, and add any methods that start with "Test" to add tests.
// Methods that do not match any suite interfaces and do not begin
// with "Test" will not be run by testify, and can safely be used as
// helper methods.
//
// Once you've built your testing suite, you need to run the suite
// (using suite.Run from testify) inside any function that matches the
// identity that "go test" is already looking for (i.e.
// func(*testing.T)).
//
// Regular expression to select test suites specified command-line
// argument "-run". Regular expression to select the methods
// of test suites specified command-line argument "-m".
// Suite object has assertion methods.
//
// A crude example:
//     // Basic imports
//     import (
//         "testing"
//         "github.com/stretchr/testify/assert"
//         "github.com/stretchr/testify/suite"
//     )
//
//     // Define the suite, and absorb the built-in basic suite
//     // functionality from testify - including a T() method which
//     // returns the current testing context
//     type ExampleTestSuite struct {
//         suite.Suite
//         VariableThatShouldStartAtFive int
//     }
//
//     // Make sure that VariableThatShouldStartAtFive is set to five
//     // before each test
//     func (suite *ExampleTestSuite) SetupTest() {
//         suite.VariableThatShouldStartAtFive = 5
//     }
//
//     // All methods that begin with "Test" are run as tests within a
//     // suite.
//     func (suite *ExampleTestSuite) TestExample() {
//         assert.Equal(suite.T(), 5, suite.VariableThatShouldStartAtFive)
//         suite.Equal(5, suite.VariableThatShouldStartAtFive)
//     }
//
//     // In order for 'go test' to run this suite, we need to create
//     // a normal test function and pass our suite to suite.Run
//     func TestExampleTestSuite(t *testing.T) {
//         suite.Run(t, new(ExampleTestSuite))
//     }
package suite
