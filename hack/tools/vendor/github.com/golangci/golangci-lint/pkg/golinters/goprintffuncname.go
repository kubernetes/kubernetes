package golinters

import (
	"github.com/jirfag/go-printf-func-name/pkg/analyzer"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewGoPrintfFuncName() *goanalysis.Linter {
	return goanalysis.NewLinter(
		"goprintffuncname",
		"Checks that printf-like functions are named with `f` at the end",
		[]*analysis.Analyzer{analyzer.Analyzer},
		nil,
	).WithLoadMode(goanalysis.LoadModeSyntax)
}
