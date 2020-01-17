/*

TeamCity Reporter for Ginkgo

Makes use of TeamCity's support for Service Messages
http://confluence.jetbrains.com/display/TCD7/Build+Script+Interaction+with+TeamCity#BuildScriptInteractionwithTeamCity-ReportingTests
*/

package reporters

import (
	"fmt"
	"io"
	"strings"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/types"
)

const (
	messageId = "##teamcity"
)

type TeamCityReporter struct {
	writer         io.Writer
	testSuiteName  string
	ReporterConfig config.DefaultReporterConfigType
}

func NewTeamCityReporter(writer io.Writer) *TeamCityReporter {
	return &TeamCityReporter{
		writer: writer,
	}
}

func (reporter *TeamCityReporter) SpecSuiteWillBegin(config config.GinkgoConfigType, summary *types.SuiteSummary) {
	reporter.testSuiteName = escape(summary.SuiteDescription)
	fmt.Fprintf(reporter.writer, "%s[testSuiteStarted name='%s']", messageId, reporter.testSuiteName)
}

func (reporter *TeamCityReporter) BeforeSuiteDidRun(setupSummary *types.SetupSummary) {
	reporter.handleSetupSummary("BeforeSuite", setupSummary)
}

func (reporter *TeamCityReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {
	reporter.handleSetupSummary("AfterSuite", setupSummary)
}

func (reporter *TeamCityReporter) handleSetupSummary(name string, setupSummary *types.SetupSummary) {
	if setupSummary.State != types.SpecStatePassed {
		testName := escape(name)
		fmt.Fprintf(reporter.writer, "%s[testStarted name='%s']", messageId, testName)
		message := escape(setupSummary.Failure.ComponentCodeLocation.String())
		details := escape(setupSummary.Failure.Message)
		fmt.Fprintf(reporter.writer, "%s[testFailed name='%s' message='%s' details='%s']", messageId, testName, message, details)
		durationInMilliseconds := setupSummary.RunTime.Seconds() * 1000
		fmt.Fprintf(reporter.writer, "%s[testFinished name='%s' duration='%v']", messageId, testName, durationInMilliseconds)
	}
}

func (reporter *TeamCityReporter) SpecWillRun(specSummary *types.SpecSummary) {
	testName := escape(strings.Join(specSummary.ComponentTexts[1:], " "))
	fmt.Fprintf(reporter.writer, "%s[testStarted name='%s']", messageId, testName)
}

func (reporter *TeamCityReporter) SpecDidComplete(specSummary *types.SpecSummary) {
	testName := escape(strings.Join(specSummary.ComponentTexts[1:], " "))

	if reporter.ReporterConfig.ReportPassed && specSummary.State == types.SpecStatePassed {
		details := escape(specSummary.CapturedOutput)
		fmt.Fprintf(reporter.writer, "%s[testPassed name='%s' details='%s']", messageId, testName, details)
	}
	if specSummary.State == types.SpecStateFailed || specSummary.State == types.SpecStateTimedOut || specSummary.State == types.SpecStatePanicked {
		message := escape(specSummary.Failure.ComponentCodeLocation.String())
		details := escape(specSummary.Failure.Message)
		fmt.Fprintf(reporter.writer, "%s[testFailed name='%s' message='%s' details='%s']", messageId, testName, message, details)
	}
	if specSummary.State == types.SpecStateSkipped || specSummary.State == types.SpecStatePending {
		fmt.Fprintf(reporter.writer, "%s[testIgnored name='%s']", messageId, testName)
	}

	durationInMilliseconds := specSummary.RunTime.Seconds() * 1000
	fmt.Fprintf(reporter.writer, "%s[testFinished name='%s' duration='%v']", messageId, testName, durationInMilliseconds)
}

func (reporter *TeamCityReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {
	fmt.Fprintf(reporter.writer, "%s[testSuiteFinished name='%s']", messageId, reporter.testSuiteName)
}

func escape(output string) string {
	output = strings.Replace(output, "|", "||", -1)
	output = strings.Replace(output, "'", "|'", -1)
	output = strings.Replace(output, "\n", "|n", -1)
	output = strings.Replace(output, "\r", "|r", -1)
	output = strings.Replace(output, "[", "|[", -1)
	output = strings.Replace(output, "]", "|]", -1)
	return output
}
