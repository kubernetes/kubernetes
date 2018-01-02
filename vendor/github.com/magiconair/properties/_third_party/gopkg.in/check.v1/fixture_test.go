// Tests for the behavior of the test fixture system.

package check_test

import (
	. "github.com/magiconair/properties/_third_party/gopkg.in/check.v1"
)

// -----------------------------------------------------------------------
// Fixture test suite.

type FixtureS struct{}

var fixtureS = Suite(&FixtureS{})

func (s *FixtureS) TestCountSuite(c *C) {
	suitesRun += 1
}

// -----------------------------------------------------------------------
// Basic fixture ordering verification.

func (s *FixtureS) TestOrder(c *C) {
	helper := FixtureHelper{}
	Run(&helper, nil)
	c.Check(helper.calls[0], Equals, "SetUpSuite")
	c.Check(helper.calls[1], Equals, "SetUpTest")
	c.Check(helper.calls[2], Equals, "Test1")
	c.Check(helper.calls[3], Equals, "TearDownTest")
	c.Check(helper.calls[4], Equals, "SetUpTest")
	c.Check(helper.calls[5], Equals, "Test2")
	c.Check(helper.calls[6], Equals, "TearDownTest")
	c.Check(helper.calls[7], Equals, "TearDownSuite")
	c.Check(len(helper.calls), Equals, 8)
}

// -----------------------------------------------------------------------
// Check the behavior when panics occur within tests and fixtures.

func (s *FixtureS) TestPanicOnTest(c *C) {
	helper := FixtureHelper{panicOn: "Test1"}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(helper.calls[0], Equals, "SetUpSuite")
	c.Check(helper.calls[1], Equals, "SetUpTest")
	c.Check(helper.calls[2], Equals, "Test1")
	c.Check(helper.calls[3], Equals, "TearDownTest")
	c.Check(helper.calls[4], Equals, "SetUpTest")
	c.Check(helper.calls[5], Equals, "Test2")
	c.Check(helper.calls[6], Equals, "TearDownTest")
	c.Check(helper.calls[7], Equals, "TearDownSuite")
	c.Check(len(helper.calls), Equals, 8)

	expected := "^\n-+\n" +
		"PANIC: check_test\\.go:[0-9]+: FixtureHelper.Test1\n\n" +
		"\\.\\.\\. Panic: Test1 \\(PC=[xA-F0-9]+\\)\n\n" +
		".+:[0-9]+\n" +
		"  in (go)?panic\n" +
		".*check_test.go:[0-9]+\n" +
		"  in FixtureHelper.trace\n" +
		".*check_test.go:[0-9]+\n" +
		"  in FixtureHelper.Test1\n" +
		"(.|\n)*$"

	c.Check(output.value, Matches, expected)
}

func (s *FixtureS) TestPanicOnSetUpTest(c *C) {
	helper := FixtureHelper{panicOn: "SetUpTest"}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(helper.calls[0], Equals, "SetUpSuite")
	c.Check(helper.calls[1], Equals, "SetUpTest")
	c.Check(helper.calls[2], Equals, "TearDownTest")
	c.Check(helper.calls[3], Equals, "TearDownSuite")
	c.Check(len(helper.calls), Equals, 4)

	expected := "^\n-+\n" +
		"PANIC: check_test\\.go:[0-9]+: " +
		"FixtureHelper\\.SetUpTest\n\n" +
		"\\.\\.\\. Panic: SetUpTest \\(PC=[xA-F0-9]+\\)\n\n" +
		".+:[0-9]+\n" +
		"  in (go)?panic\n" +
		".*check_test.go:[0-9]+\n" +
		"  in FixtureHelper.trace\n" +
		".*check_test.go:[0-9]+\n" +
		"  in FixtureHelper.SetUpTest\n" +
		"(.|\n)*" +
		"\n-+\n" +
		"PANIC: check_test\\.go:[0-9]+: " +
		"FixtureHelper\\.Test1\n\n" +
		"\\.\\.\\. Panic: Fixture has panicked " +
		"\\(see related PANIC\\)\n$"

	c.Check(output.value, Matches, expected)
}

func (s *FixtureS) TestPanicOnTearDownTest(c *C) {
	helper := FixtureHelper{panicOn: "TearDownTest"}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(helper.calls[0], Equals, "SetUpSuite")
	c.Check(helper.calls[1], Equals, "SetUpTest")
	c.Check(helper.calls[2], Equals, "Test1")
	c.Check(helper.calls[3], Equals, "TearDownTest")
	c.Check(helper.calls[4], Equals, "TearDownSuite")
	c.Check(len(helper.calls), Equals, 5)

	expected := "^\n-+\n" +
		"PANIC: check_test\\.go:[0-9]+: " +
		"FixtureHelper.TearDownTest\n\n" +
		"\\.\\.\\. Panic: TearDownTest \\(PC=[xA-F0-9]+\\)\n\n" +
		".+:[0-9]+\n" +
		"  in (go)?panic\n" +
		".*check_test.go:[0-9]+\n" +
		"  in FixtureHelper.trace\n" +
		".*check_test.go:[0-9]+\n" +
		"  in FixtureHelper.TearDownTest\n" +
		"(.|\n)*" +
		"\n-+\n" +
		"PANIC: check_test\\.go:[0-9]+: " +
		"FixtureHelper\\.Test1\n\n" +
		"\\.\\.\\. Panic: Fixture has panicked " +
		"\\(see related PANIC\\)\n$"

	c.Check(output.value, Matches, expected)
}

func (s *FixtureS) TestPanicOnSetUpSuite(c *C) {
	helper := FixtureHelper{panicOn: "SetUpSuite"}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(helper.calls[0], Equals, "SetUpSuite")
	c.Check(helper.calls[1], Equals, "TearDownSuite")
	c.Check(len(helper.calls), Equals, 2)

	expected := "^\n-+\n" +
		"PANIC: check_test\\.go:[0-9]+: " +
		"FixtureHelper.SetUpSuite\n\n" +
		"\\.\\.\\. Panic: SetUpSuite \\(PC=[xA-F0-9]+\\)\n\n" +
		".+:[0-9]+\n" +
		"  in (go)?panic\n" +
		".*check_test.go:[0-9]+\n" +
		"  in FixtureHelper.trace\n" +
		".*check_test.go:[0-9]+\n" +
		"  in FixtureHelper.SetUpSuite\n" +
		"(.|\n)*$"

	c.Check(output.value, Matches, expected)
}

func (s *FixtureS) TestPanicOnTearDownSuite(c *C) {
	helper := FixtureHelper{panicOn: "TearDownSuite"}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(helper.calls[0], Equals, "SetUpSuite")
	c.Check(helper.calls[1], Equals, "SetUpTest")
	c.Check(helper.calls[2], Equals, "Test1")
	c.Check(helper.calls[3], Equals, "TearDownTest")
	c.Check(helper.calls[4], Equals, "SetUpTest")
	c.Check(helper.calls[5], Equals, "Test2")
	c.Check(helper.calls[6], Equals, "TearDownTest")
	c.Check(helper.calls[7], Equals, "TearDownSuite")
	c.Check(len(helper.calls), Equals, 8)

	expected := "^\n-+\n" +
		"PANIC: check_test\\.go:[0-9]+: " +
		"FixtureHelper.TearDownSuite\n\n" +
		"\\.\\.\\. Panic: TearDownSuite \\(PC=[xA-F0-9]+\\)\n\n" +
		".+:[0-9]+\n" +
		"  in (go)?panic\n" +
		".*check_test.go:[0-9]+\n" +
		"  in FixtureHelper.trace\n" +
		".*check_test.go:[0-9]+\n" +
		"  in FixtureHelper.TearDownSuite\n" +
		"(.|\n)*$"

	c.Check(output.value, Matches, expected)
}

// -----------------------------------------------------------------------
// A wrong argument on a test or fixture will produce a nice error.

func (s *FixtureS) TestPanicOnWrongTestArg(c *C) {
	helper := WrongTestArgHelper{}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(helper.calls[0], Equals, "SetUpSuite")
	c.Check(helper.calls[1], Equals, "SetUpTest")
	c.Check(helper.calls[2], Equals, "TearDownTest")
	c.Check(helper.calls[3], Equals, "SetUpTest")
	c.Check(helper.calls[4], Equals, "Test2")
	c.Check(helper.calls[5], Equals, "TearDownTest")
	c.Check(helper.calls[6], Equals, "TearDownSuite")
	c.Check(len(helper.calls), Equals, 7)

	expected := "^\n-+\n" +
		"PANIC: fixture_test\\.go:[0-9]+: " +
		"WrongTestArgHelper\\.Test1\n\n" +
		"\\.\\.\\. Panic: WrongTestArgHelper\\.Test1 argument " +
		"should be \\*check\\.C\n"

	c.Check(output.value, Matches, expected)
}

func (s *FixtureS) TestPanicOnWrongSetUpTestArg(c *C) {
	helper := WrongSetUpTestArgHelper{}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(len(helper.calls), Equals, 0)

	expected :=
		"^\n-+\n" +
			"PANIC: fixture_test\\.go:[0-9]+: " +
			"WrongSetUpTestArgHelper\\.SetUpTest\n\n" +
			"\\.\\.\\. Panic: WrongSetUpTestArgHelper\\.SetUpTest argument " +
			"should be \\*check\\.C\n"

	c.Check(output.value, Matches, expected)
}

func (s *FixtureS) TestPanicOnWrongSetUpSuiteArg(c *C) {
	helper := WrongSetUpSuiteArgHelper{}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(len(helper.calls), Equals, 0)

	expected :=
		"^\n-+\n" +
			"PANIC: fixture_test\\.go:[0-9]+: " +
			"WrongSetUpSuiteArgHelper\\.SetUpSuite\n\n" +
			"\\.\\.\\. Panic: WrongSetUpSuiteArgHelper\\.SetUpSuite argument " +
			"should be \\*check\\.C\n"

	c.Check(output.value, Matches, expected)
}

// -----------------------------------------------------------------------
// Nice errors also when tests or fixture have wrong arg count.

func (s *FixtureS) TestPanicOnWrongTestArgCount(c *C) {
	helper := WrongTestArgCountHelper{}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(helper.calls[0], Equals, "SetUpSuite")
	c.Check(helper.calls[1], Equals, "SetUpTest")
	c.Check(helper.calls[2], Equals, "TearDownTest")
	c.Check(helper.calls[3], Equals, "SetUpTest")
	c.Check(helper.calls[4], Equals, "Test2")
	c.Check(helper.calls[5], Equals, "TearDownTest")
	c.Check(helper.calls[6], Equals, "TearDownSuite")
	c.Check(len(helper.calls), Equals, 7)

	expected := "^\n-+\n" +
		"PANIC: fixture_test\\.go:[0-9]+: " +
		"WrongTestArgCountHelper\\.Test1\n\n" +
		"\\.\\.\\. Panic: WrongTestArgCountHelper\\.Test1 argument " +
		"should be \\*check\\.C\n"

	c.Check(output.value, Matches, expected)
}

func (s *FixtureS) TestPanicOnWrongSetUpTestArgCount(c *C) {
	helper := WrongSetUpTestArgCountHelper{}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(len(helper.calls), Equals, 0)

	expected :=
		"^\n-+\n" +
			"PANIC: fixture_test\\.go:[0-9]+: " +
			"WrongSetUpTestArgCountHelper\\.SetUpTest\n\n" +
			"\\.\\.\\. Panic: WrongSetUpTestArgCountHelper\\.SetUpTest argument " +
			"should be \\*check\\.C\n"

	c.Check(output.value, Matches, expected)
}

func (s *FixtureS) TestPanicOnWrongSetUpSuiteArgCount(c *C) {
	helper := WrongSetUpSuiteArgCountHelper{}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(len(helper.calls), Equals, 0)

	expected :=
		"^\n-+\n" +
			"PANIC: fixture_test\\.go:[0-9]+: " +
			"WrongSetUpSuiteArgCountHelper\\.SetUpSuite\n\n" +
			"\\.\\.\\. Panic: WrongSetUpSuiteArgCountHelper" +
			"\\.SetUpSuite argument should be \\*check\\.C\n"

	c.Check(output.value, Matches, expected)
}

// -----------------------------------------------------------------------
// Helper test suites with wrong function arguments.

type WrongTestArgHelper struct {
	FixtureHelper
}

func (s *WrongTestArgHelper) Test1(t int) {
}

type WrongSetUpTestArgHelper struct {
	FixtureHelper
}

func (s *WrongSetUpTestArgHelper) SetUpTest(t int) {
}

type WrongSetUpSuiteArgHelper struct {
	FixtureHelper
}

func (s *WrongSetUpSuiteArgHelper) SetUpSuite(t int) {
}

type WrongTestArgCountHelper struct {
	FixtureHelper
}

func (s *WrongTestArgCountHelper) Test1(c *C, i int) {
}

type WrongSetUpTestArgCountHelper struct {
	FixtureHelper
}

func (s *WrongSetUpTestArgCountHelper) SetUpTest(c *C, i int) {
}

type WrongSetUpSuiteArgCountHelper struct {
	FixtureHelper
}

func (s *WrongSetUpSuiteArgCountHelper) SetUpSuite(c *C, i int) {
}

// -----------------------------------------------------------------------
// Ensure fixture doesn't run without tests.

type NoTestsHelper struct {
	hasRun bool
}

func (s *NoTestsHelper) SetUpSuite(c *C) {
	s.hasRun = true
}

func (s *NoTestsHelper) TearDownSuite(c *C) {
	s.hasRun = true
}

func (s *FixtureS) TestFixtureDoesntRunWithoutTests(c *C) {
	helper := NoTestsHelper{}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Check(helper.hasRun, Equals, false)
}

// -----------------------------------------------------------------------
// Verify that checks and assertions work correctly inside the fixture.

type FixtureCheckHelper struct {
	fail      string
	completed bool
}

func (s *FixtureCheckHelper) SetUpSuite(c *C) {
	switch s.fail {
	case "SetUpSuiteAssert":
		c.Assert(false, Equals, true)
	case "SetUpSuiteCheck":
		c.Check(false, Equals, true)
	}
	s.completed = true
}

func (s *FixtureCheckHelper) SetUpTest(c *C) {
	switch s.fail {
	case "SetUpTestAssert":
		c.Assert(false, Equals, true)
	case "SetUpTestCheck":
		c.Check(false, Equals, true)
	}
	s.completed = true
}

func (s *FixtureCheckHelper) Test(c *C) {
	// Do nothing.
}

func (s *FixtureS) TestSetUpSuiteCheck(c *C) {
	helper := FixtureCheckHelper{fail: "SetUpSuiteCheck"}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Assert(output.value, Matches,
		"\n---+\n"+
			"FAIL: fixture_test\\.go:[0-9]+: "+
			"FixtureCheckHelper\\.SetUpSuite\n\n"+
			"fixture_test\\.go:[0-9]+:\n"+
			"    c\\.Check\\(false, Equals, true\\)\n"+
			"\\.+ obtained bool = false\n"+
			"\\.+ expected bool = true\n\n")
	c.Assert(helper.completed, Equals, true)
}

func (s *FixtureS) TestSetUpSuiteAssert(c *C) {
	helper := FixtureCheckHelper{fail: "SetUpSuiteAssert"}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Assert(output.value, Matches,
		"\n---+\n"+
			"FAIL: fixture_test\\.go:[0-9]+: "+
			"FixtureCheckHelper\\.SetUpSuite\n\n"+
			"fixture_test\\.go:[0-9]+:\n"+
			"    c\\.Assert\\(false, Equals, true\\)\n"+
			"\\.+ obtained bool = false\n"+
			"\\.+ expected bool = true\n\n")
	c.Assert(helper.completed, Equals, false)
}

// -----------------------------------------------------------------------
// Verify that logging within SetUpTest() persists within the test log itself.

type FixtureLogHelper struct {
	c *C
}

func (s *FixtureLogHelper) SetUpTest(c *C) {
	s.c = c
	c.Log("1")
}

func (s *FixtureLogHelper) Test(c *C) {
	c.Log("2")
	s.c.Log("3")
	c.Log("4")
	c.Fail()
}

func (s *FixtureLogHelper) TearDownTest(c *C) {
	s.c.Log("5")
}

func (s *FixtureS) TestFixtureLogging(c *C) {
	helper := FixtureLogHelper{}
	output := String{}
	Run(&helper, &RunConf{Output: &output})
	c.Assert(output.value, Matches,
		"\n---+\n"+
			"FAIL: fixture_test\\.go:[0-9]+: "+
			"FixtureLogHelper\\.Test\n\n"+
			"1\n2\n3\n4\n5\n")
}

// -----------------------------------------------------------------------
// Skip() within fixture methods.

func (s *FixtureS) TestSkipSuite(c *C) {
	helper := FixtureHelper{skip: true, skipOnN: 0}
	output := String{}
	result := Run(&helper, &RunConf{Output: &output})
	c.Assert(output.value, Equals, "")
	c.Assert(helper.calls[0], Equals, "SetUpSuite")
	c.Assert(helper.calls[1], Equals, "TearDownSuite")
	c.Assert(len(helper.calls), Equals, 2)
	c.Assert(result.Skipped, Equals, 2)
}

func (s *FixtureS) TestSkipTest(c *C) {
	helper := FixtureHelper{skip: true, skipOnN: 1}
	output := String{}
	result := Run(&helper, &RunConf{Output: &output})
	c.Assert(helper.calls[0], Equals, "SetUpSuite")
	c.Assert(helper.calls[1], Equals, "SetUpTest")
	c.Assert(helper.calls[2], Equals, "SetUpTest")
	c.Assert(helper.calls[3], Equals, "Test2")
	c.Assert(helper.calls[4], Equals, "TearDownTest")
	c.Assert(helper.calls[5], Equals, "TearDownSuite")
	c.Assert(len(helper.calls), Equals, 6)
	c.Assert(result.Skipped, Equals, 1)
}
