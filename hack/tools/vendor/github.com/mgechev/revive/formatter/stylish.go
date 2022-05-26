package formatter

import (
	"bytes"
	"fmt"

	"github.com/fatih/color"
	"github.com/mgechev/revive/lint"
	"github.com/olekukonko/tablewriter"
)

// Stylish is an implementation of the Formatter interface
// which formats the errors to JSON.
type Stylish struct {
	Metadata lint.FormatterMetadata
}

// Name returns the name of the formatter
func (f *Stylish) Name() string {
	return "stylish"
}

func formatFailure(failure lint.Failure, severity lint.Severity) []string {
	fString := color.CyanString(failure.Failure)
	fName := color.RedString("https://revive.run/r#" + failure.RuleName)
	lineColumn := failure.Position
	pos := fmt.Sprintf("(%d, %d)", lineColumn.Start.Line, lineColumn.Start.Column)
	if severity == lint.SeverityWarning {
		fName = color.YellowString("https://revive.run/r#" + failure.RuleName)
	}
	return []string{failure.GetFilename(), pos, fName, fString}
}

// Format formats the failures gotten from the lint.
func (f *Stylish) Format(failures <-chan lint.Failure, config lint.Config) (string, error) {
	var result [][]string
	var totalErrors = 0
	var total = 0

	for f := range failures {
		total++
		currentType := severity(config, f)
		if currentType == lint.SeverityError {
			totalErrors++
		}
		result = append(result, formatFailure(f, lint.Severity(currentType)))
	}
	ps := "problems"
	if total == 1 {
		ps = "problem"
	}

	fileReport := make(map[string][][]string)

	for _, row := range result {
		if _, ok := fileReport[row[0]]; !ok {
			fileReport[row[0]] = [][]string{}
		}

		fileReport[row[0]] = append(fileReport[row[0]], []string{row[1], row[2], row[3]})
	}

	output := ""
	for filename, val := range fileReport {
		buf := new(bytes.Buffer)
		table := tablewriter.NewWriter(buf)
		table.SetBorder(false)
		table.SetColumnSeparator("")
		table.SetRowSeparator("")
		table.SetAutoWrapText(false)
		table.AppendBulk(val)
		table.Render()
		c := color.New(color.Underline)
		output += c.SprintfFunc()(filename + "\n")
		output += buf.String() + "\n"
	}

	suffix := fmt.Sprintf(" %d %s (%d errors) (%d warnings)", total, ps, totalErrors, total-totalErrors)

	if total > 0 && totalErrors > 0 {
		suffix = color.RedString("\n ✖" + suffix)
	} else if total > 0 && totalErrors == 0 {
		suffix = color.YellowString("\n ✖" + suffix)
	} else {
		suffix, output = "", ""
	}

	return output + suffix, nil
}
