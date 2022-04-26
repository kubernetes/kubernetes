package formatter

import (
	"bytes"
	"fmt"
	"sort"

	"github.com/fatih/color"
	"github.com/mgechev/revive/lint"
	"github.com/olekukonko/tablewriter"
)

var newLines = map[rune]bool{
	0x000A: true,
	0x000B: true,
	0x000C: true,
	0x000D: true,
	0x0085: true,
	0x2028: true,
	0x2029: true,
}

func getErrorEmoji() string {
	return color.RedString("✘")
}

func getWarningEmoji() string {
	return color.YellowString("⚠")
}

// Friendly is an implementation of the Formatter interface
// which formats the errors to JSON.
type Friendly struct {
	Metadata lint.FormatterMetadata
}

// Name returns the name of the formatter
func (f *Friendly) Name() string {
	return "friendly"
}

// Format formats the failures gotten from the lint.
func (f *Friendly) Format(failures <-chan lint.Failure, config lint.Config) (string, error) {
	errorMap := map[string]int{}
	warningMap := map[string]int{}
	totalErrors := 0
	totalWarnings := 0
	for failure := range failures {
		sev := severity(config, failure)
		f.printFriendlyFailure(failure, sev)
		if sev == lint.SeverityWarning {
			warningMap[failure.RuleName]++
			totalWarnings++
		}
		if sev == lint.SeverityError {
			errorMap[failure.RuleName]++
			totalErrors++
		}
	}
	f.printSummary(totalErrors, totalWarnings)
	f.printStatistics(color.RedString("Errors:"), errorMap)
	f.printStatistics(color.YellowString("Warnings:"), warningMap)
	return "", nil
}

func (f *Friendly) printFriendlyFailure(failure lint.Failure, severity lint.Severity) {
	f.printHeaderRow(failure, severity)
	f.printFilePosition(failure)
	fmt.Println()
	fmt.Println()
}

func (f *Friendly) printHeaderRow(failure lint.Failure, severity lint.Severity) {
	emoji := getWarningEmoji()
	if severity == lint.SeverityError {
		emoji = getErrorEmoji()
	}
	fmt.Print(f.table([][]string{{emoji, "https://revive.run/r#" + failure.RuleName, color.GreenString(failure.Failure)}}))
}

func (f *Friendly) printFilePosition(failure lint.Failure) {
	fmt.Printf("  %s:%d:%d", failure.GetFilename(), failure.Position.Start.Line, failure.Position.Start.Column)
}

type statEntry struct {
	name     string
	failures int
}

func (f *Friendly) printSummary(errors, warnings int) {
	emoji := getWarningEmoji()
	if errors > 0 {
		emoji = getErrorEmoji()
	}
	problemsLabel := "problems"
	if errors+warnings == 1 {
		problemsLabel = "problem"
	}
	warningsLabel := "warnings"
	if warnings == 1 {
		warningsLabel = "warning"
	}
	errorsLabel := "errors"
	if errors == 1 {
		errorsLabel = "error"
	}
	str := fmt.Sprintf("%d %s (%d %s, %d %s)", errors+warnings, problemsLabel, errors, errorsLabel, warnings, warningsLabel)
	if errors > 0 {
		fmt.Printf("%s %s\n", emoji, color.RedString(str))
		fmt.Println()
		return
	}
	if warnings > 0 {
		fmt.Printf("%s %s\n", emoji, color.YellowString(str))
		fmt.Println()
		return
	}
}

func (f *Friendly) printStatistics(header string, stats map[string]int) {
	if len(stats) == 0 {
		return
	}
	var data []statEntry
	for name, total := range stats {
		data = append(data, statEntry{name, total})
	}
	sort.Slice(data, func(i, j int) bool {
		return data[i].failures > data[j].failures
	})
	formatted := [][]string{}
	for _, entry := range data {
		formatted = append(formatted, []string{color.GreenString(fmt.Sprintf("%d", entry.failures)), entry.name})
	}
	fmt.Println(header)
	fmt.Println(f.table(formatted))
}

func (f *Friendly) table(rows [][]string) string {
	buf := new(bytes.Buffer)
	table := tablewriter.NewWriter(buf)
	table.SetBorder(false)
	table.SetColumnSeparator("")
	table.SetRowSeparator("")
	table.SetAutoWrapText(false)
	table.AppendBulk(rows)
	table.Render()
	return buf.String()
}
