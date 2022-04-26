package goanalysis

import (
	"go/token"

	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/result"
)

type Issue struct {
	result.Issue
	Pass *analysis.Pass
}

func NewIssue(i *result.Issue, pass *analysis.Pass) Issue {
	return Issue{
		Issue: *i,
		Pass:  pass,
	}
}

type EncodingIssue struct {
	FromLinter           string
	Text                 string
	Pos                  token.Position
	LineRange            *result.Range
	Replacement          *result.Replacement
	ExpectNoLint         bool
	ExpectedNoLintLinter string
}
