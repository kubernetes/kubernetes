// A set of packages that provide many tools for testifying that your code will behave as you intend.
//
// testify contains the following packages:
//
// The assert package provides a comprehensive set of assertion functions that tie in to the Go testing system.
//
// The http package contains tools to make it easier to test http activity using the Go testing system.
//
// The mock package provides a system by which it is possible to mock your objects and verify calls are happening as expected.
//
// The suite package provides a basic structure for using structs as testing suites, and methods on those structs as tests.  It includes setup/teardown functionality in the way of interfaces.
package testify

import (
	_ "github.com/stretchr/testify/assert"
	_ "github.com/stretchr/testify/http"
	_ "github.com/stretchr/testify/mock"
)
