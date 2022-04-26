package golinters

import (
	"github.com/moricho/tparallel"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewTparallel() *goanalysis.Linter {
	analyzers := []*analysis.Analyzer{
		tparallel.Analyzer,
	}

	return goanalysis.NewLinter(
		"tparallel",
		"tparallel detects inappropriate usage of t.Parallel() method in your Go test codes",
		analyzers,
		nil,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
