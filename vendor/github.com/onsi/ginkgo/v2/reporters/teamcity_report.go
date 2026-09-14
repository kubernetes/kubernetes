/*

TeamCity Reporter for Ginkgo

Makes use of TeamCity's support for Service Messages
http://confluence.jetbrains.com/display/TCD7/Build+Script+Interaction+with+TeamCity#BuildScriptInteractionwithTeamCity-ReportingTests
*/

package reporters

import (
	"fmt"
	"os"
	"path"
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
	if err := os.MkdirAll(path.Dir(dst), 0770); err != nil {
		return err
	}
	f, err := os.Create(dst)
	if err != nil {
		return err
	}

	name := report.SuiteDescription
	labels := report.SuiteLabels
	semVerConstraints := report.SuiteSemVerConstraints
	componentSemVerConstraints := report.SuiteComponentSemVerConstraints
	if len(labels) > 0 {
		name = name + " [" + strings.Join(labels, ", ") + "]"
	}
	if len(semVerConstraints) > 0 {
		name = name + " [" + strings.Join(semVerConstraints, ", ") + "]"
	}
	if len(componentSemVerConstraints) > 0 {
		name = name + " [" + formatComponentSemVerConstraintsToString(componentSemVerConstraints) + "]"
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
		semVerConstraints := spec.SemVerConstraints()
		if len(semVerConstraints) > 0 {
			name = name + " [" + strings.Join(semVerConstraints, ", ") + "]"
		}
		componentSemVerConstraints := spec.ComponentSemVerConstraints()
		if len(componentSemVerConstraints) > 0 {
			name = name + " [" + formatComponentSemVerConstraintsToString(componentSemVerConstraints) + "]"
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
			details := failureDescriptionForUnstructuredReporters(spec)
			fmt.Fprintf(f, "##teamcity[testFailed name='%s' message='failed - %s' details='%s']\n", name, tcEscape(spec.Failure.Message), tcEscape(details))
		case types.SpecStatePanicked:
			details := failureDescriptionForUnstructuredReporters(spec)
			fmt.Fprintf(f, "##teamcity[testFailed name='%s' message='panicked - %s' details='%s']\n", name, tcEscape(spec.Failure.ForwardedPanic), tcEscape(details))
		case types.SpecStateTimedout:
			details := failureDescriptionForUnstructuredReporters(spec)
			fmt.Fprintf(f, "##teamcity[testFailed name='%s' message='timedout - %s' details='%s']\n", name, tcEscape(spec.Failure.Message), tcEscape(details))
		case types.SpecStateInterrupted:
			details := failureDescriptionForUnstructuredReporters(spec)
			fmt.Fprintf(f, "##teamcity[testFailed name='%s' message='interrupted - %s' details='%s']\n", name, tcEscape(spec.Failure.Message), tcEscape(details))
		case types.SpecStateAborted:
			details := failureDescriptionForUnstructuredReporters(spec)
			fmt.Fprintf(f, "##teamcity[testFailed name='%s' message='aborted - %s' details='%s']\n", name, tcEscape(spec.Failure.Message), tcEscape(details))
		}

		fmt.Fprintf(f, "##teamcity[testStdOut name='%s' out='%s']\n", name, tcEscape(systemOutForUnstructuredReporters(spec)))
		fmt.Fprintf(f, "##teamcity[testStdErr name='%s' out='%s']\n", name, tcEscape(systemErrForUnstructuredReporters(spec)))
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
