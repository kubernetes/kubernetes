package golinters

import (
	"github.com/kunwardeep/paralleltest/pkg/paralleltest"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewParallelTest() *goanalysis.Linter {
	analyzers := []*analysis.Analyzer{
		paralleltest.NewAnalyzer(),
	}

	return goanalysis.NewLinter(
		"paralleltest",
		"paralleltest detects missing usage of t.Parallel() method in your Go test",
		analyzers,
		nil,
	).WithLoadMode(goanalysis.LoadModeSyntax)
}
