/*

JUnit XML Reporter for Ginkgo

For usage instructions: http://onsi.github.io/ginkgo/#generating_junit_xml_output

*/

package reporters

import (
	"encoding/xml"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/types"
)

type JUnitTestSuite struct {
	XMLName   xml.Name        `xml:"testsuite"`
	TestCases []JUnitTestCase `xml:"testcase"`
	Name      string          `xml:"name,attr"`
	Tests     int             `xml:"tests,attr"`
	Failures  int             `xml:"failures,attr"`
	Errors    int             `xml:"errors,attr"`
	Time      float64         `xml:"time,attr"`
}

type JUnitTestCase struct {
	Name           string               `xml:"name,attr"`
	ClassName      string               `xml:"classname,attr"`
	PassedMessage  *JUnitPassedMessage  `xml:"passed,omitempty"`
	FailureMessage *JUnitFailureMessage `xml:"failure,omitempty"`
	Skipped        *JUnitSkipped        `xml:"skipped,omitempty"`
	Time           float64              `xml:"time,attr"`
	SystemOut      string               `xml:"system-out,omitempty"`
}

type JUnitPassedMessage struct {
	Message string `xml:",chardata"`
}

type JUnitFailureMessage struct {
	Type    string `xml:"type,attr"`
	Message string `xml:",chardata"`
}

type JUnitSkipped struct {
	Message string `xml:",chardata"`
}

type JUnitReporter struct {
	suite          JUnitTestSuite
	filename       string
	testSuiteName  string
	ReporterConfig config.DefaultReporterConfigType
}

//NewJUnitReporter creates a new JUnit XML reporter.  The XML will be stored in the passed in filename.
func NewJUnitReporter(filename string) *JUnitReporter {
	return &JUnitReporter{
		filename: filename,
	}
}

func (reporter *JUnitReporter) SpecSuiteWillBegin(ginkgoConfig config.GinkgoConfigType, summary *types.SuiteSummary) {
	reporter.suite = JUnitTestSuite{
		Name:      summary.SuiteDescription,
		TestCases: []JUnitTestCase{},
	}
	reporter.testSuiteName = summary.SuiteDescription
	reporter.ReporterConfig = config.DefaultReporterConfig
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
	if reporter.ReporterConfig.ReportPassed && specSummary.State == types.SpecStatePassed {
		testCase.PassedMessage = &JUnitPassedMessage{
			Message: specSummary.CapturedOutput,
		}
	}
	if specSummary.State == types.SpecStateFailed || specSummary.State == types.SpecStateTimedOut || specSummary.State == types.SpecStatePanicked {
		testCase.FailureMessage = &JUnitFailureMessage{
			Type:    reporter.failureTypeForState(specSummary.State),
			Message: failureMessage(specSummary.Failure),
		}
		if specSummary.State == types.SpecStatePanicked {
			testCase.FailureMessage.Message += fmt.Sprintf("\n\nPanic: %s\n\nFull stack:\n%s",
				specSummary.Failure.ForwardedPanic,
				specSummary.Failure.Location.FullStackTrace)
		}
		testCase.SystemOut = specSummary.CapturedOutput
	}
	if specSummary.State == types.SpecStateSkipped || specSummary.State == types.SpecStatePending {
		testCase.Skipped = &JUnitSkipped{}
		if specSummary.Failure.Message != "" {
			testCase.Skipped.Message = failureMessage(specSummary.Failure)
		}
	}
	testCase.Time = specSummary.RunTime.Seconds()
	reporter.suite.TestCases = append(reporter.suite.TestCases, testCase)
}

func (reporter *JUnitReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {
	reporter.suite.Tests = summary.NumberOfSpecsThatWillBeRun
	reporter.suite.Time = math.Trunc(summary.RunTime.Seconds()*1000) / 1000
	reporter.suite.Failures = summary.NumberOfFailedSpecs
	reporter.suite.Errors = 0
	if reporter.ReporterConfig.ReportFile != "" {
		reporter.filename = reporter.ReporterConfig.ReportFile
		fmt.Printf("\nJUnit path was configured: %s\n", reporter.filename)
	}
	filePath, _ := filepath.Abs(reporter.filename)
	dirPath := filepath.Dir(filePath)
	err := os.MkdirAll(dirPath, os.ModePerm)
	if err != nil {
		fmt.Printf("\nFailed to create JUnit directory: %s\n\t%s", filePath, err.Error())
	}
	file, err := os.Create(filePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create JUnit report file: %s\n\t%s", filePath, err.Error())
	}
	defer file.Close()
	file.WriteString(xml.Header)
	encoder := xml.NewEncoder(file)
	encoder.Indent("  ", "    ")
	err = encoder.Encode(reporter.suite)
	if err == nil {
		fmt.Fprintf(os.Stdout, "\nJUnit report was created: %s\n", filePath)
	} else {
		fmt.Fprintf(os.Stderr,"\nFailed to generate JUnit report data:\n\t%s", err.Error())
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
