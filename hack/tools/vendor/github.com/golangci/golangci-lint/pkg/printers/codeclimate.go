package printers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/golangci/golangci-lint/pkg/result"
)

// CodeClimateIssue is a subset of the Code Climate spec - https://github.com/codeclimate/spec/blob/master/SPEC.md#data-types
// It is just enough to support GitLab CI Code Quality - https://docs.gitlab.com/ee/user/project/merge_requests/code_quality.html
type CodeClimateIssue struct {
	Description string `json:"description"`
	Severity    string `json:"severity,omitempty"`
	Fingerprint string `json:"fingerprint"`
	Location    struct {
		Path  string `json:"path"`
		Lines struct {
			Begin int `json:"begin"`
		} `json:"lines"`
	} `json:"location"`
}

type CodeClimate struct {
	w io.Writer
}

func NewCodeClimate(w io.Writer) *CodeClimate {
	return &CodeClimate{w: w}
}

func (p CodeClimate) Print(ctx context.Context, issues []result.Issue) error {
	codeClimateIssues := make([]CodeClimateIssue, 0, len(issues))
	for i := range issues {
		issue := &issues[i]
		codeClimateIssue := CodeClimateIssue{}
		codeClimateIssue.Description = issue.Description()
		codeClimateIssue.Location.Path = issue.Pos.Filename
		codeClimateIssue.Location.Lines.Begin = issue.Pos.Line
		codeClimateIssue.Fingerprint = issue.Fingerprint()

		if issue.Severity != "" {
			codeClimateIssue.Severity = issue.Severity
		}

		codeClimateIssues = append(codeClimateIssues, codeClimateIssue)
	}

	outputJSON, err := json.Marshal(codeClimateIssues)
	if err != nil {
		return err
	}

	_, err = fmt.Fprint(p.w, string(outputJSON))
	if err != nil {
		return err
	}
	return nil
}
