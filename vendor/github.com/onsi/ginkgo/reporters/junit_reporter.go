/*

JUnit XML Reporter for Ginkgo

For usage instructions: http://onsi.github.io/ginkgo/#generating_junit_xml_output

*/

package reporters

import (
	"encoding/xml"
	"fmt"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/types"
	"os"
	"strings"
)

type JUnitTestSuite struct {
	XMLName   xml.Name        `xml:"testsuite"`
	TestCases []JUnitTestCase `xml:"testcase"`
	Tests     int             `xml:"tests,attr"`
	Failures  int             `xml:"failures,attr"`
	Time      float64         `xml:"time,attr"`
}

type JUnitTestCase struct {
	Name           string               `xml:"name,attr"`
	ClassName      string               `xml:"classname,attr"`
	FailureMessage *JUnitFailureMessage `xml:"failure,omitempty"`
	Skipped        *JUnitSkipped        `xml:"skipped,omitempty"`
	Time           float64              `xml:"time,attr"`
	SystemOut      string               `xml:"system-out,omitempty"`
}

type JUnitFailureMessage struct {
	Type    string `xml:"type,attr"`
	Message string `xml:",chardata"`
}

type JUnitSkipped struct {
	XMLName xml.Name `xml:"skipped"`
}

type JUnitReporter struct {
	suite         JUnitTestSuite
	filename      string
	testSuiteName string
}

//NewJUnitReporter creates a new JUnit XML reporter.  The XML will be stored in the passed in filename.
func NewJUnitReporter(filename string) *JUnitReporter {
	return &JUnitReporter{
		filename: filename,
	}
}

func (reporter *JUnitReporter) SpecSuiteWillBegin(config config.GinkgoConfigType, summary *types.SuiteSummary) {
	reporter.suite = JUnitTestSuite{
		Tests:     summary.NumberOfSpecsThatWillBeRun,
		TestCases: []JUnitTestCase{},
	}
	reporter.testSuiteName = summary.SuiteDescription
}

func (reporter *JUnitReporter) SpecWillRun(specSummary *types.SpecSummary) {
}

func (reporter *JUnitReporter) BeforeSuiteDidRun(setupSummary *types.SetupSummary) {
	reporter.handleSetupSummary("BeforeSuite", setupSummary)
}

func (reporter *JUnitReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {
	reporter.handleSetupSummary("AfterSuite", setupSummary)
}

func failureMessage(failure types.SpecFailure) string {
	return fmt.Sprintf("%s\n%s\n%s", failure.ComponentCodeLocation.String(), failure.Message, failure.Location.String())
}

func (reporter *JUnitReporter) handleSetupSummary(name string, setupSummary *types.SetupSummary) {
	if setupSummary.State != types.SpecStatePassed {
		testCase := JUnitTestCase{
			Name:      name,
			ClassName: reporter.testSuiteName,
		}

		testCase.FailureMessage = &JUnitFailureMessage{
			Type:    reporter.failureTypeForState(setupSummary.State),
			Message: failureMessage(setupSummary.Failure),
		}
		testCase.SystemOut = setupSummary.CapturedOutput
		testCase.Time = setupSummary.RunTime.Seconds()
		reporter.suite.TestCases = append(reporter.suite.TestCases, testCase)
	}
}

func (reporter *JUnitReporter) SpecDidComplete(specSummary *types.SpecSummary) {
	testCase := JUnitTestCase{
		Name:      strings.Join(specSummary.ComponentTexts[1:], " "),
		ClassName: reporter.testSuiteName,
	}
	if specSummary.State == types.SpecStateFailed || specSummary.State == types.SpecStateTimedOut || specSummary.State == types.SpecStatePanicked {
		testCase.FailureMessage = &JUnitFailureMessage{
			Type:    reporter.failureTypeForState(specSummary.State),
			Message: failureMessage(specSummary.Failure),
		}
		testCase.SystemOut = specSummary.CapturedOutput
	}
	if specSummary.State == types.SpecStateSkipped || specSummary.State == types.SpecStatePending {
		testCase.Skipped = &JUnitSkipped{}
	}
	testCase.Time = specSummary.RunTime.Seconds()
	reporter.suite.TestCases = append(reporter.suite.TestCases, testCase)
}

func (reporter *JUnitReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {
	reporter.suite.Time = summary.RunTime.Seconds()
	reporter.suite.Failures = summary.NumberOfFailedSpecs
	file, err := os.Create(reporter.filename)
	if err != nil {
		fmt.Printf("Failed to create JUnit report file: %s\n\t%s", reporter.filename, err.Error())
	}
	defer file.Close()
	file.WriteString(xml.Header)
	encoder := xml.NewEncoder(file)
	encoder.Indent("  ", "    ")
	err = encoder.Encode(reporter.suite)
	if err != nil {
		fmt.Printf("Failed to generate JUnit report\n\t%s", err.Error())
	}
}

func (reporter *JUnitReporter) failureTypeForState(state types.SpecState) string {
	switch state {
	case types.SpecStateFailed:
		return "Failure"
	case types.SpecStateTimedOut:
		return "Timeout"
	case types.SpecStatePanicked:
		return "Panic"
	default:
		return ""
	}
}
