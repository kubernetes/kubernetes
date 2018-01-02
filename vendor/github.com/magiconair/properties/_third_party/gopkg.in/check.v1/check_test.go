// This file contains just a few generic helpers which are used by the
// other test files.

package check_test

import (
	"flag"
	"fmt"
	"os"
	"regexp"
	"runtime"
	"testing"
	"time"

	"github.com/magiconair/properties/_third_party/gopkg.in/check.v1"
)

// We count the number of suites run at least to get a vague hint that the
// test suite is behaving as it should.  Otherwise a bug introduced at the
// very core of the system could go unperceived.
const suitesRunExpected = 8

var suitesRun int = 0

func Test(t *testing.T) {
	check.TestingT(t)
	if suitesRun != suitesRunExpected && flag.Lookup("check.f").Value.String() == "" {
		critical(fmt.Sprintf("Expected %d suites to run rather than %d",
			suitesRunExpected, suitesRun))
	}
}

// -----------------------------------------------------------------------
// Helper functions.

// Break down badly.  This is used in test cases which can't yet assume
// that the fundamental bits are working.
func critical(error string) {
	fmt.Fprintln(os.Stderr, "CRITICAL: "+error)
	os.Exit(1)
}

// Return the file line where it's called.
func getMyLine() int {
	if _, _, line, ok := runtime.Caller(1); ok {
		return line
	}
	return -1
}

// -----------------------------------------------------------------------
// Helper type implementing a basic io.Writer for testing output.

// Type implementing the io.Writer interface for analyzing output.
type String struct {
	value string
}

// The only function required by the io.Writer interface.  Will append
// written data to the String.value string.
func (s *String) Write(p []byte) (n int, err error) {
	s.value += string(p)
	return len(p), nil
}

// Trivial wrapper to test errors happening on a different file
// than the test itself.
func checkEqualWrapper(c *check.C, obtained, expected interface{}) (result bool, line int) {
	return c.Check(obtained, check.Equals, expected), getMyLine()
}

// -----------------------------------------------------------------------
// Helper suite for testing basic fail behavior.

type FailHelper struct {
	testLine int
}

func (s *FailHelper) TestLogAndFail(c *check.C) {
	s.testLine = getMyLine() - 1
	c.Log("Expected failure!")
	c.Fail()
}

// -----------------------------------------------------------------------
// Helper suite for testing basic success behavior.

type SuccessHelper struct{}

func (s *SuccessHelper) TestLogAndSucceed(c *check.C) {
	c.Log("Expected success!")
}

// -----------------------------------------------------------------------
// Helper suite for testing ordering and behavior of fixture.

type FixtureHelper struct {
	calls   []string
	panicOn string
	skip    bool
	skipOnN int
	sleepOn string
	sleep   time.Duration
	bytes   int64
}

func (s *FixtureHelper) trace(name string, c *check.C) {
	s.calls = append(s.calls, name)
	if name == s.panicOn {
		panic(name)
	}
	if s.sleep > 0 && s.sleepOn == name {
		time.Sleep(s.sleep)
	}
	if s.skip && s.skipOnN == len(s.calls)-1 {
		c.Skip("skipOnN == n")
	}
}

func (s *FixtureHelper) SetUpSuite(c *check.C) {
	s.trace("SetUpSuite", c)
}

func (s *FixtureHelper) TearDownSuite(c *check.C) {
	s.trace("TearDownSuite", c)
}

func (s *FixtureHelper) SetUpTest(c *check.C) {
	s.trace("SetUpTest", c)
}

func (s *FixtureHelper) TearDownTest(c *check.C) {
	s.trace("TearDownTest", c)
}

func (s *FixtureHelper) Test1(c *check.C) {
	s.trace("Test1", c)
}

func (s *FixtureHelper) Test2(c *check.C) {
	s.trace("Test2", c)
}

func (s *FixtureHelper) Benchmark1(c *check.C) {
	s.trace("Benchmark1", c)
	for i := 0; i < c.N; i++ {
		time.Sleep(s.sleep)
	}
}

func (s *FixtureHelper) Benchmark2(c *check.C) {
	s.trace("Benchmark2", c)
	c.SetBytes(1024)
	for i := 0; i < c.N; i++ {
		time.Sleep(s.sleep)
	}
}

func (s *FixtureHelper) Benchmark3(c *check.C) {
	var x []int64
	s.trace("Benchmark3", c)
	for i := 0; i < c.N; i++ {
		time.Sleep(s.sleep)
		x = make([]int64, 5)
		_ = x
	}
}

// -----------------------------------------------------------------------
// Helper which checks the state of the test and ensures that it matches
// the given expectations.  Depends on c.Errorf() working, so shouldn't
// be used to test this one function.

type expectedState struct {
	name   string
	result interface{}
	failed bool
	log    string
}

// Verify the state of the test.  Note that since this also verifies if
// the test is supposed to be in a failed state, no other checks should
// be done in addition to what is being tested.
func checkState(c *check.C, result interface{}, expected *expectedState) {
	failed := c.Failed()
	c.Succeed()
	log := c.GetTestLog()
	matched, matchError := regexp.MatchString("^"+expected.log+"$", log)
	if matchError != nil {
		c.Errorf("Error in matching expression used in testing %s",
			expected.name)
	} else if !matched {
		c.Errorf("%s logged:\n----------\n%s----------\n\nExpected:\n----------\n%s\n----------",
			expected.name, log, expected.log)
	}
	if result != expected.result {
		c.Errorf("%s returned %#v rather than %#v",
			expected.name, result, expected.result)
	}
	if failed != expected.failed {
		if failed {
			c.Errorf("%s has failed when it shouldn't", expected.name)
		} else {
			c.Errorf("%s has not failed when it should", expected.name)
		}
	}
}
