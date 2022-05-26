package golinters

import (
	"github.com/Antonboom/errname/pkg/analyzer"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewErrName() *goanalysis.Linter {
	analyzers := []*analysis.Analyzer{
		analyzer.New(),
	}

	return goanalysis.NewLinter(
		"errname",
		"Checks that sentinel errors are prefixed with the `Err` and error types are suffixed with the `Error`.",
		analyzers,
		nil,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
