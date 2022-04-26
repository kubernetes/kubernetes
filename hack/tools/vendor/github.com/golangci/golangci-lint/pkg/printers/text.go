package printers

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/fatih/color"

	"github.com/golangci/golangci-lint/pkg/logutils"
	"github.com/golangci/golangci-lint/pkg/result"
)

type Text struct {
	printIssuedLine bool
	useColors       bool
	printLinterName bool

	log logutils.Log
	w   io.Writer
}

func NewText(printIssuedLine, useColors, printLinterName bool, log logutils.Log, w io.Writer) *Text {
	return &Text{
		printIssuedLine: printIssuedLine,
		useColors:       useColors,
		printLinterName: printLinterName,
		log:             log,
		w:               w,
	}
}

func (p Text) SprintfColored(ca color.Attribute, format string, args ...interface{}) string {
	if !p.useColors {
		return fmt.Sprintf(format, args...)
	}

	c := color.New(ca)
	return c.Sprintf(format, args...)
}

func (p *Text) Print(ctx context.Context, issues []result.Issue) error {
	for i := range issues {
		p.printIssue(&issues[i])

		if !p.printIssuedLine {
			continue
		}

		p.printSourceCode(&issues[i])
		p.printUnderLinePointer(&issues[i])
	}

	return nil
}

func (p Text) printIssue(i *result.Issue) {
	text := p.SprintfColored(color.FgRed, "%s", strings.TrimSpace(i.Text))
	if p.printLinterName {
		text += fmt.Sprintf(" (%s)", i.FromLinter)
	}
	pos := p.SprintfColored(color.Bold, "%s:%d", i.FilePath(), i.Line())
	if i.Pos.Column != 0 {
		pos += fmt.Sprintf(":%d", i.Pos.Column)
	}
	fmt.Fprintf(p.w, "%s: %s\n", pos, text)
}

func (p Text) printSourceCode(i *result.Issue) {
	for _, line := range i.SourceLines {
		fmt.Fprintln(p.w, line)
	}
}

func (p Text) printUnderLinePointer(i *result.Issue) {
	// if column == 0 it means column is unknown (e.g. for gosec)
	if len(i.SourceLines) != 1 || i.Pos.Column == 0 {
		return
	}

	col0 := i.Pos.Column - 1
	line := i.SourceLines[0]
	prefixRunes := make([]rune, 0, len(line))
	for j := 0; j < len(line) && j < col0; j++ {
		if line[j] == '\t' {
			prefixRunes = append(prefixRunes, '\t')
		} else {
			prefixRunes = append(prefixRunes, ' ')
		}
	}

	fmt.Fprintf(p.w, "%s%s\n", string(prefixRunes), p.SprintfColored(color.FgYellow, "^"))
}
