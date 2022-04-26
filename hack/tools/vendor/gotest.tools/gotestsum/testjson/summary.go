package testjson

import (
	"fmt"
	"io"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/fatih/color"
)

// Summary enumerates the sections which can be printed by PrintSummary
type Summary int

// nolint: golint
const (
	SummarizeNone    Summary = 0
	SummarizeSkipped Summary = (1 << iota) / 2
	SummarizeFailed
	SummarizeErrors
	SummarizeOutput
	SummarizeAll = SummarizeSkipped | SummarizeFailed | SummarizeErrors | SummarizeOutput
)

var summaryValues = map[Summary]string{
	SummarizeSkipped: "skipped",
	SummarizeFailed:  "failed",
	SummarizeErrors:  "errors",
	SummarizeOutput:  "output",
}

var summaryFromValue = map[string]Summary{
	"none":    SummarizeNone,
	"skipped": SummarizeSkipped,
	"failed":  SummarizeFailed,
	"errors":  SummarizeErrors,
	"output":  SummarizeOutput,
	"all":     SummarizeAll,
}

func (s Summary) String() string {
	if s == SummarizeNone {
		return "none"
	}
	var result []string
	for v := Summary(1); v <= s; v <<= 1 {
		if s.Includes(v) {
			result = append(result, summaryValues[v])
		}
	}
	return strings.Join(result, ",")
}

// Includes returns true if Summary includes all the values set by other.
func (s Summary) Includes(other Summary) bool {
	return s&other == other
}

// NewSummary returns a new Summary from a string value. If the string does not
// match any known values returns false for the second value.
func NewSummary(value string) (Summary, bool) {
	s, ok := summaryFromValue[value]
	return s, ok
}

// PrintSummary of a test Execution. Prints a section for each summary type
// followed by a DONE line to out.
func PrintSummary(out io.Writer, execution *Execution, opts Summary) {
	execSummary := newExecSummary(execution, opts)
	if opts.Includes(SummarizeSkipped) {
		writeTestCaseSummary(out, execSummary, formatSkipped())
	}
	if opts.Includes(SummarizeFailed) {
		writeTestCaseSummary(out, execSummary, formatFailed())
	}

	errors := execution.Errors()
	if opts.Includes(SummarizeErrors) {
		writeErrorSummary(out, errors)
	}

	fmt.Fprintf(out, "\n%s %d tests%s%s%s in %s\n",
		formatExecStatus(execution),
		execution.Total(),
		formatTestCount(len(execution.Skipped()), "skipped", ""),
		formatTestCount(len(execution.Failed()), "failure", "s"),
		formatTestCount(countErrors(errors), "error", "s"),
		FormatDurationAsSeconds(execution.Elapsed(), 3))
}

func formatTestCount(count int, category string, pluralize string) string {
	switch count {
	case 0:
		return ""
	case 1:
	default:
		category += pluralize
	}
	return fmt.Sprintf(", %d %s", count, category)
}

func formatExecStatus(exec *Execution) string {
	if !exec.done {
		return ""
	}
	var runs string
	if exec.lastRunID > 0 {
		runs = fmt.Sprintf(" %d runs,", exec.lastRunID+1)
	}
	return "DONE" + runs
}

// FormatDurationAsSeconds formats a time.Duration as a float with an s suffix.
func FormatDurationAsSeconds(d time.Duration, precision int) string {
	if d == neverFinished {
		return "unknown"
	}
	return fmt.Sprintf("%.[2]*[1]fs", d.Seconds(), precision)
}

func writeErrorSummary(out io.Writer, errors []string) {
	if len(errors) > 0 {
		fmt.Fprintln(out, color.MagentaString("\n=== Errors"))
	}
	for _, err := range errors {
		fmt.Fprintln(out, err)
	}
}

// countErrors in stderr lines. Build errors may include multiple lines where
// subsequent lines are indented.
// FIXME: Panics will include multiple lines, and are still overcounted.
func countErrors(errors []string) int {
	var count int
	for _, line := range errors {
		r, _ := utf8.DecodeRuneInString(line)
		if !unicode.IsSpace(r) {
			count++
		}
	}
	return count
}

type executionSummary interface {
	Failed() []TestCase
	Skipped() []TestCase
	OutputLines(TestCase) []string
}

type noOutputSummary struct {
	*Execution
}

func (s *noOutputSummary) OutputLines(_ TestCase) []string {
	return nil
}

func newExecSummary(execution *Execution, opts Summary) executionSummary {
	if opts.Includes(SummarizeOutput) {
		return execution
	}
	return &noOutputSummary{Execution: execution}
}

func writeTestCaseSummary(out io.Writer, execution executionSummary, conf testCaseFormatConfig) {
	testCases := conf.getter(execution)
	if len(testCases) == 0 {
		return
	}
	fmt.Fprintln(out, "\n=== "+conf.header)
	for idx, tc := range testCases {
		fmt.Fprintf(out, "=== %s: %s %s%s (%s)\n",
			conf.prefix,
			RelativePackagePath(tc.Package),
			tc.Test,
			formatRunID(tc.RunID),
			FormatDurationAsSeconds(tc.Elapsed, 2))
		for _, line := range execution.OutputLines(tc) {
			if isFramingLine(line) || conf.filter(tc.Test.Name(), line) {
				continue
			}
			fmt.Fprint(out, line)
		}
		if _, isNoOutput := execution.(*noOutputSummary); !isNoOutput && idx+1 != len(testCases) {
			fmt.Fprintln(out)
		}
	}
}

type testCaseFormatConfig struct {
	header string
	prefix string
	filter func(testName string, line string) bool
	getter func(executionSummary) []TestCase
}

func formatFailed() testCaseFormatConfig {
	withColor := color.RedString
	return testCaseFormatConfig{
		header: withColor("Failed"),
		prefix: withColor("FAIL"),
		filter: func(testName string, line string) bool {
			return strings.HasPrefix(line, "--- FAIL: "+testName+" ")
		},
		getter: func(execution executionSummary) []TestCase {
			return execution.Failed()
		},
	}
}

func formatSkipped() testCaseFormatConfig {
	withColor := color.YellowString
	return testCaseFormatConfig{
		header: withColor("Skipped"),
		prefix: withColor("SKIP"),
		filter: func(testName string, line string) bool {
			return strings.HasPrefix(line, "--- SKIP: "+testName+" ")
		},
		getter: func(execution executionSummary) []TestCase {
			return execution.Skipped()
		},
	}
}

func isFramingLine(line string) bool {
	return strings.HasPrefix(line, "=== RUN   Test") ||
		strings.HasPrefix(line, "=== PAUSE Test") ||
		strings.HasPrefix(line, "=== CONT  Test")
}
