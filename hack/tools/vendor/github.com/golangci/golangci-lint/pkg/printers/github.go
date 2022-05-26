package printers

import (
	"context"
	"fmt"
	"io"

	"github.com/golangci/golangci-lint/pkg/result"
)

type github struct {
	w io.Writer
}

const defaultGithubSeverity = "error"

// NewGithub output format outputs issues according to GitHub actions format:
// https://help.github.com/en/actions/reference/workflow-commands-for-github-actions#setting-an-error-message
func NewGithub(w io.Writer) Printer {
	return &github{w: w}
}

// print each line as: ::error file=app.js,line=10,col=15::Something went wrong
func formatIssueAsGithub(issue *result.Issue) string {
	severity := defaultGithubSeverity
	if issue.Severity != "" {
		severity = issue.Severity
	}

	ret := fmt.Sprintf("::%s file=%s,line=%d", severity, issue.FilePath(), issue.Line())
	if issue.Pos.Column != 0 {
		ret += fmt.Sprintf(",col=%d", issue.Pos.Column)
	}

	ret += fmt.Sprintf("::%s (%s)", issue.Text, issue.FromLinter)
	return ret
}

func (p *github) Print(_ context.Context, issues []result.Issue) error {
	for ind := range issues {
		_, err := fmt.Fprintln(p.w, formatIssueAsGithub(&issues[ind]))
		if err != nil {
			return err
		}
	}
	return nil
}
