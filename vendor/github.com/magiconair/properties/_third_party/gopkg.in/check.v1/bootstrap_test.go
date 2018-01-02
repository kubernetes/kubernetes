// These initial tests are for bootstrapping.  They verify that we can
// basically use the testing infrastructure itself to check if the test
// system is working.
//
// These tests use will break down the test runner badly in case of
// errors because if they simply fail, we can't be sure the developer
// will ever see anything (because failing means the failing system
// somehow isn't working! :-)
//
// Do not assume *any* internal functionality works as expected besides
// what's actually tested here.

package check_test

import (
	"fmt"
	"github.com/magiconair/properties/_third_party/gopkg.in/check.v1"
	"strings"
)

type BootstrapS struct{}

var boostrapS = check.Suite(&BootstrapS{})

func (s *BootstrapS) TestCountSuite(c *check.C) {
	suitesRun += 1
}

func (s *BootstrapS) TestFailedAndFail(c *check.C) {
	if c.Failed() {
		critical("c.Failed() must be false first!")
	}
	c.Fail()
	if !c.Failed() {
		critical("c.Fail() didn't put the test in a failed state!")
	}
	c.Succeed()
}

func (s *BootstrapS) TestFailedAndSucceed(c *check.C) {
	c.Fail()
	c.Succeed()
	if c.Failed() {
		critical("c.Succeed() didn't put the test back in a non-failed state")
	}
}

func (s *BootstrapS) TestLogAndGetTestLog(c *check.C) {
	c.Log("Hello there!")
	log := c.GetTestLog()
	if log != "Hello there!\n" {
		critical(fmt.Sprintf("Log() or GetTestLog() is not working! Got: %#v", log))
	}
}

func (s *BootstrapS) TestLogfAndGetTestLog(c *check.C) {
	c.Logf("Hello %v", "there!")
	log := c.GetTestLog()
	if log != "Hello there!\n" {
		critical(fmt.Sprintf("Logf() or GetTestLog() is not working! Got: %#v", log))
	}
}

func (s *BootstrapS) TestRunShowsErrors(c *check.C) {
	output := String{}
	check.Run(&FailHelper{}, &check.RunConf{Output: &output})
	if strings.Index(output.value, "Expected failure!") == -1 {
		critical(fmt.Sprintf("RunWithWriter() output did not contain the "+
			"expected failure! Got: %#v",
			output.value))
	}
}

func (s *BootstrapS) TestRunDoesntShowSuccesses(c *check.C) {
	output := String{}
	check.Run(&SuccessHelper{}, &check.RunConf{Output: &output})
	if strings.Index(output.value, "Expected success!") != -1 {
		critical(fmt.Sprintf("RunWithWriter() output contained a successful "+
			"test! Got: %#v",
			output.value))
	}
}
