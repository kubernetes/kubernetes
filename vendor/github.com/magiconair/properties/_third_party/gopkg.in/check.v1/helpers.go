package check

import (
	"fmt"
	"strings"
	"time"
)

// TestName returns the current test name in the form "SuiteName.TestName"
func (c *C) TestName() string {
	return c.testName
}

// -----------------------------------------------------------------------
// Basic succeeding/failing logic.

// Failed returns whether the currently running test has already failed.
func (c *C) Failed() bool {
	return c.status == failedSt
}

// Fail marks the currently running test as failed.
//
// Something ought to have been previously logged so the developer can tell
// what went wrong. The higher level helper functions will fail the test
// and do the logging properly.
func (c *C) Fail() {
	c.status = failedSt
}

// FailNow marks the currently running test as failed and stops running it.
// Something ought to have been previously logged so the developer can tell
// what went wrong. The higher level helper functions will fail the test
// and do the logging properly.
func (c *C) FailNow() {
	c.Fail()
	c.stopNow()
}

// Succeed marks the currently running test as succeeded, undoing any
// previous failures.
func (c *C) Succeed() {
	c.status = succeededSt
}

// SucceedNow marks the currently running test as succeeded, undoing any
// previous failures, and stops running the test.
func (c *C) SucceedNow() {
	c.Succeed()
	c.stopNow()
}

// ExpectFailure informs that the running test is knowingly broken for
// the provided reason. If the test does not fail, an error will be reported
// to raise attention to this fact. This method is useful to temporarily
// disable tests which cover well known problems until a better time to
// fix the problem is found, without forgetting about the fact that a
// failure still exists.
func (c *C) ExpectFailure(reason string) {
	if reason == "" {
		panic("Missing reason why the test is expected to fail")
	}
	c.mustFail = true
	c.reason = reason
}

// Skip skips the running test for the provided reason. If run from within
// SetUpTest, the individual test being set up will be skipped, and if run
// from within SetUpSuite, the whole suite is skipped.
func (c *C) Skip(reason string) {
	if reason == "" {
		panic("Missing reason why the test is being skipped")
	}
	c.reason = reason
	c.status = skippedSt
	c.stopNow()
}

// -----------------------------------------------------------------------
// Basic logging.

// GetTestLog returns the current test error output.
func (c *C) GetTestLog() string {
	return c.logb.String()
}

// Log logs some information into the test error output.
// The provided arguments are assembled together into a string with fmt.Sprint.
func (c *C) Log(args ...interface{}) {
	c.log(args...)
}

// Log logs some information into the test error output.
// The provided arguments are assembled together into a string with fmt.Sprintf.
func (c *C) Logf(format string, args ...interface{}) {
	c.logf(format, args...)
}

// Output enables *C to be used as a logger in functions that require only
// the minimum interface of *log.Logger.
func (c *C) Output(calldepth int, s string) error {
	d := time.Now().Sub(c.startTime)
	msec := d / time.Millisecond
	sec := d / time.Second
	min := d / time.Minute

	c.Logf("[LOG] %d:%02d.%03d %s", min, sec%60, msec%1000, s)
	return nil
}

// Error logs an error into the test error output and marks the test as failed.
// The provided arguments are assembled together into a string with fmt.Sprint.
func (c *C) Error(args ...interface{}) {
	c.logCaller(1)
	c.logString(fmt.Sprint("Error: ", fmt.Sprint(args...)))
	c.logNewLine()
	c.Fail()
}

// Errorf logs an error into the test error output and marks the test as failed.
// The provided arguments are assembled together into a string with fmt.Sprintf.
func (c *C) Errorf(format string, args ...interface{}) {
	c.logCaller(1)
	c.logString(fmt.Sprintf("Error: "+format, args...))
	c.logNewLine()
	c.Fail()
}

// Fatal logs an error into the test error output, marks the test as failed, and
// stops the test execution. The provided arguments are assembled together into
// a string with fmt.Sprint.
func (c *C) Fatal(args ...interface{}) {
	c.logCaller(1)
	c.logString(fmt.Sprint("Error: ", fmt.Sprint(args...)))
	c.logNewLine()
	c.FailNow()
}

// Fatlaf logs an error into the test error output, marks the test as failed, and
// stops the test execution. The provided arguments are assembled together into
// a string with fmt.Sprintf.
func (c *C) Fatalf(format string, args ...interface{}) {
	c.logCaller(1)
	c.logString(fmt.Sprint("Error: ", fmt.Sprintf(format, args...)))
	c.logNewLine()
	c.FailNow()
}

// -----------------------------------------------------------------------
// Generic checks and assertions based on checkers.

// Check verifies if the first value matches the expected value according
// to the provided checker. If they do not match, an error is logged, the
// test is marked as failed, and the test execution continues.
//
// Some checkers may not need the expected argument (e.g. IsNil).
//
// Extra arguments provided to the function are logged next to the reported
// problem when the matching fails.
func (c *C) Check(obtained interface{}, checker Checker, args ...interface{}) bool {
	return c.internalCheck("Check", obtained, checker, args...)
}

// Assert ensures that the first value matches the expected value according
// to the provided checker. If they do not match, an error is logged, the
// test is marked as failed, and the test execution stops.
//
// Some checkers may not need the expected argument (e.g. IsNil).
//
// Extra arguments provided to the function are logged next to the reported
// problem when the matching fails.
func (c *C) Assert(obtained interface{}, checker Checker, args ...interface{}) {
	if !c.internalCheck("Assert", obtained, checker, args...) {
		c.stopNow()
	}
}

func (c *C) internalCheck(funcName string, obtained interface{}, checker Checker, args ...interface{}) bool {
	if checker == nil {
		c.logCaller(2)
		c.logString(fmt.Sprintf("%s(obtained, nil!?, ...):", funcName))
		c.logString("Oops.. you've provided a nil checker!")
		c.logNewLine()
		c.Fail()
		return false
	}

	// If the last argument is a bug info, extract it out.
	var comment CommentInterface
	if len(args) > 0 {
		if c, ok := args[len(args)-1].(CommentInterface); ok {
			comment = c
			args = args[:len(args)-1]
		}
	}

	params := append([]interface{}{obtained}, args...)
	info := checker.Info()

	if len(params) != len(info.Params) {
		names := append([]string{info.Params[0], info.Name}, info.Params[1:]...)
		c.logCaller(2)
		c.logString(fmt.Sprintf("%s(%s):", funcName, strings.Join(names, ", ")))
		c.logString(fmt.Sprintf("Wrong number of parameters for %s: want %d, got %d", info.Name, len(names), len(params)+1))
		c.logNewLine()
		c.Fail()
		return false
	}

	// Copy since it may be mutated by Check.
	names := append([]string{}, info.Params...)

	// Do the actual check.
	result, error := checker.Check(params, names)
	if !result || error != "" {
		c.logCaller(2)
		for i := 0; i != len(params); i++ {
			c.logValue(names[i], params[i])
		}
		if comment != nil {
			c.logString(comment.CheckCommentString())
		}
		if error != "" {
			c.logString(error)
		}
		c.logNewLine()
		c.Fail()
		return false
	}
	return true
}
