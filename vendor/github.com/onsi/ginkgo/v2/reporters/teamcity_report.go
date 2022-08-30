/*

TeamCity Reporter for Ginkgo

Makes use of TeamCity's support for Service Messages
http://confluence.jetbrains.com/display/TCD7/Build+Script+Interaction+with+TeamCity#BuildScriptInteractionwithTeamCity-ReportingTests
*/

package reporters

import (
	"fmt"
	"os"
	"strings"

	"github.com/onsi/ginkgo/v2/types"
)

func tcEscape(s string) string {
	s = strings.ReplaceAll(s, "|", "||")
	s = strings.ReplaceAll(s, "'", "|'")
	s = strings.ReplaceAll(s, "\n", "|n")
	s = strings.ReplaceAll(s, "\r", "|r")
	s = strings.ReplaceAll(s, "[", "|[")
	s = strings.ReplaceAll(s, "]", "|]")
	return s
}

func GenerateTeamcityReport(report types.Report, dst string) error {
	f, err := os.Create(dst)
	if err != nil {
		return err
	}

	name := report.SuiteDescription
	labels := report.SuiteLabels
	if len(labels) > 0 {
		name = name + " [" + strings.Join(labels, ", ") + "]"
	}
	fmt.Fprintf(f, "##teamcity[testSuiteStarted name='%s']\n", tcEscape(name))
	for _, spec := range report.SpecReports {
		name := fmt.Sprintf("[%s]", spec.LeafNodeType)
		if spec.FullText() != "" {
			name = name + " " + spec.FullText()
		}
		labels := spec.Labels()
		if len(labels) > 0 {
			name = name + " [" + strings.Join(labels, ", ") + "]"
		}

		name = tcEscape(name)
		fmt.Fprintf(f, "##teamcity[testStarted name='%s']\n", name)
		switch spec.State {
		case types.SpecStatePending:
			fmt.Fprintf(f, "##teamcity[testIgnored name='%s' message='pending']\n", name)
		case types.SpecStateSkipped:
			message := "skipped"
			if spec.Failure.Message != "" {
				message += " - " + spec.Failure.Message
			}
			fmt.Fprintf(f, "##teamcity[testIgnored name='%s' message='%s']\n", name, tcEscape(message))
		case types.SpecStateFailed:
			details := fmt.Sprintf("%s\n%s", spec.Failure.Location.String(), spec.Failure.Location.FullStackTrace)
			fmt.Fprintf(f, "##teamcity[testFailed name='%s' message='failed - %s' details='%s']\n", name, tcEscape(spec.Failure.Message), tcEscape(details))
		case types.SpecStatePanicked:
			details := fmt.Sprintf("%s\n%s", spec.Failure.Location.String(), spec.Failure.Location.FullStackTrace)
			fmt.Fprintf(f, "##teamcity[testFailed name='%s' message='panicked - %s' details='%s']\n", name, tcEscape(spec.Failure.ForwardedPanic), tcEscape(details))
		case types.SpecStateInterrupted:
			fmt.Fprintf(f, "##teamcity[testFailed name='%s' message='interrupted' details='%s']\n", name, tcEscape(spec.Failure.Message))
		case types.SpecStateAborted:
			details := fmt.Sprintf("%s\n%s", spec.Failure.Location.String(), spec.Failure.Location.FullStackTrace)
			fmt.Fprintf(f, "##teamcity[testFailed name='%s' message='aborted - %s' details='%s']\n", name, tcEscape(spec.Failure.Message), tcEscape(details))
		}

		fmt.Fprintf(f, "##teamcity[testStdOut name='%s' out='%s']\n", name, tcEscape(systemOutForUnstructureReporters(spec)))
		fmt.Fprintf(f, "##teamcity[testStdErr name='%s' out='%s']\n", name, tcEscape(spec.CapturedGinkgoWriterOutput))
		fmt.Fprintf(f, "##teamcity[testFinished name='%s' duration='%d']\n", name, int(spec.RunTime.Seconds()*1000.0))
	}
	fmt.Fprintf(f, "##teamcity[testSuiteFinished name='%s']\n", tcEscape(report.SuiteDescription))

	return f.Close()
}

func MergeAndCleanupTeamcityReports(sources []string, dst string) ([]string, error) {
	messages := []string{}
	merged := []byte{}
	for _, source := range sources {
		data, err := os.ReadFile(source)
		if err != nil {
			messages = append(messages, fmt.Sprintf("Could not open %s:\n%s", source, err.Error()))
			continue
		}
		os.Remove(source)
		merged = append(merged, data...)
	}
	return messages, os.WriteFile(dst, merged, 0666)
}
