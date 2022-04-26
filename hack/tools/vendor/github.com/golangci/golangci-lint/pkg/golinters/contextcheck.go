package golinters

import (
	"github.com/sylvia7788/contextcheck"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewContextCheck() *goanalysis.Linter {
	analyzer := contextcheck.NewAnalyzer()
	return goanalysis.NewLinter(
		"contextcheck",
		"check the function whether use a non-inherited context",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
