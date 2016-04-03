package suite

import (
	"errors"
	"io/ioutil"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

// SuiteRequireTwice is intended to test the usage of suite.Require in two
// different tests
type SuiteRequireTwice struct{ Suite }

// TestSuiteRequireTwice checks for regressions of issue #149 where
// suite.requirements was not initialised in suite.SetT()
// A regression would result on these tests panicking rather than failing.
func TestSuiteRequireTwice(t *testing.T) {
	ok := testing.RunTests(
		func(_, _ string) (bool, error) { return true, nil },
		[]testing.InternalTest{{
			Name: "TestSuiteRequireTwice",
			F: func(t *testing.T) {
				suite := new(SuiteRequireTwice)
				Run(t, suite)
			},
		}},
	)
	assert.Equal(t, false, ok)
}

func (s *SuiteRequireTwice) TestRequireOne() {
	r := s.Require()
	r.Equal(1, 2)
}

func (s *SuiteRequireTwice) TestRequireTwo() {
	r := s.Require()
	r.Equal(1, 2)
}

// This suite is intended to store values to make sure that only
// testing-suite-related methods are run.  It's also a fully
// functional example of a testing suite, using setup/teardown methods
// and a helper method that is ignored by testify.  To make this look
// more like a real world example, all tests in the suite perform some
// type of assertion.
type SuiteTester struct {
	// Include our basic suite logic.
	Suite

	// Keep counts of how many times each method is run.
	SetupSuiteRunCount    int
	TearDownSuiteRunCount int
	SetupTestRunCount     int
	TearDownTestRunCount  int
	TestOneRunCount       int
	TestTwoRunCount       int
	NonTestMethodRunCount int
}

type SuiteSkipTester struct {
	// Include our basic suite logic.
	Suite

	// Keep counts of how many times each method is run.
	SetupSuiteRunCount    int
	TearDownSuiteRunCount int
}

// The SetupSuite method will be run by testify once, at the very
// start of the testing suite, before any tests are run.
func (suite *SuiteTester) SetupSuite() {
	suite.SetupSuiteRunCount++
}

func (suite *SuiteSkipTester) SetupSuite() {
	suite.SetupSuiteRunCount++
	suite.T().Skip()
}

// The TearDownSuite method will be run by testify once, at the very
// end of the testing suite, after all tests have been run.
func (suite *SuiteTester) TearDownSuite() {
	suite.TearDownSuiteRunCount++
}

func (suite *SuiteSkipTester) TearDownSuite() {
	suite.TearDownSuiteRunCount++
}

// The SetupTest method will be run before every test in the suite.
func (suite *SuiteTester) SetupTest() {
	suite.SetupTestRunCount++
}

// The TearDownTest method will be run after every test in the suite.
func (suite *SuiteTester) TearDownTest() {
	suite.TearDownTestRunCount++
}

// Every method in a testing suite that begins with "Test" will be run
// as a test.  TestOne is an example of a test.  For the purposes of
// this example, we've included assertions in the tests, since most
// tests will issue assertions.
func (suite *SuiteTester) TestOne() {
	beforeCount := suite.TestOneRunCount
	suite.TestOneRunCount++
	assert.Equal(suite.T(), suite.TestOneRunCount, beforeCount+1)
	suite.Equal(suite.TestOneRunCount, beforeCount+1)
}

// TestTwo is another example of a test.
func (suite *SuiteTester) TestTwo() {
	beforeCount := suite.TestTwoRunCount
	suite.TestTwoRunCount++
	assert.NotEqual(suite.T(), suite.TestTwoRunCount, beforeCount)
	suite.NotEqual(suite.TestTwoRunCount, beforeCount)
}

func (suite *SuiteTester) TestSkip() {
	suite.T().Skip()
}

// NonTestMethod does not begin with "Test", so it will not be run by
// testify as a test in the suite.  This is useful for creating helper
// methods for your tests.
func (suite *SuiteTester) NonTestMethod() {
	suite.NonTestMethodRunCount++
}

// TestRunSuite will be run by the 'go test' command, so within it, we
// can run our suite using the Run(*testing.T, TestingSuite) function.
func TestRunSuite(t *testing.T) {
	suiteTester := new(SuiteTester)
	Run(t, suiteTester)

	// Normally, the test would end here.  The following are simply
	// some assertions to ensure that the Run function is working as
	// intended - they are not part of the example.

	// The suite was only run once, so the SetupSuite and TearDownSuite
	// methods should have each been run only once.
	assert.Equal(t, suiteTester.SetupSuiteRunCount, 1)
	assert.Equal(t, suiteTester.TearDownSuiteRunCount, 1)

	// There are three test methods (TestOne, TestTwo, and TestSkip), so
	// the SetupTest and TearDownTest methods (which should be run once for
	// each test) should have been run three times.
	assert.Equal(t, suiteTester.SetupTestRunCount, 3)
	assert.Equal(t, suiteTester.TearDownTestRunCount, 3)

	// Each test should have been run once.
	assert.Equal(t, suiteTester.TestOneRunCount, 1)
	assert.Equal(t, suiteTester.TestTwoRunCount, 1)

	// Methods that don't match the test method identifier shouldn't
	// have been run at all.
	assert.Equal(t, suiteTester.NonTestMethodRunCount, 0)

	suiteSkipTester := new(SuiteSkipTester)
	Run(t, suiteSkipTester)

	// The suite was only run once, so the SetupSuite and TearDownSuite
	// methods should have each been run only once, even though SetupSuite
	// called Skip()
	assert.Equal(t, suiteSkipTester.SetupSuiteRunCount, 1)
	assert.Equal(t, suiteSkipTester.TearDownSuiteRunCount, 1)

}

func TestSuiteGetters(t *testing.T) {
	suite := new(SuiteTester)
	suite.SetT(t)
	assert.NotNil(t, suite.Assert())
	assert.Equal(t, suite.Assertions, suite.Assert())
	assert.NotNil(t, suite.Require())
	assert.Equal(t, suite.require, suite.Require())
}

type SuiteLoggingTester struct {
	Suite
}

func (s *SuiteLoggingTester) TestLoggingPass() {
	s.T().Log("TESTLOGPASS")
}

func (s *SuiteLoggingTester) TestLoggingFail() {
	s.T().Log("TESTLOGFAIL")
	assert.NotNil(s.T(), nil) // expected to fail
}

type StdoutCapture struct {
	oldStdout *os.File
	readPipe  *os.File
}

func (sc *StdoutCapture) StartCapture() {
	sc.oldStdout = os.Stdout
	sc.readPipe, os.Stdout, _ = os.Pipe()
}

func (sc *StdoutCapture) StopCapture() (string, error) {
	if sc.oldStdout == nil || sc.readPipe == nil {
		return "", errors.New("StartCapture not called before StopCapture")
	}
	os.Stdout.Close()
	os.Stdout = sc.oldStdout
	bytes, err := ioutil.ReadAll(sc.readPipe)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

func TestSuiteLogging(t *testing.T) {
	testT := testing.T{}

	suiteLoggingTester := new(SuiteLoggingTester)

	capture := StdoutCapture{}
	capture.StartCapture()
	Run(&testT, suiteLoggingTester)
	output, err := capture.StopCapture()

	assert.Nil(t, err, "Got an error trying to capture stdout!")

	// Failed tests' output is always printed
	assert.Contains(t, output, "TESTLOGFAIL")

	if testing.Verbose() {
		// In verbose mode, output from successful tests is also printed
		assert.Contains(t, output, "TESTLOGPASS")
	} else {
		assert.NotContains(t, output, "TESTLOGPASS")
	}
}
