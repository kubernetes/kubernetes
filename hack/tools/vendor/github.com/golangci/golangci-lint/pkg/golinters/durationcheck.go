package golinters

import (
	"github.com/charithe/durationcheck"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewDurationCheck() *goanalysis.Linter {
	a := durationcheck.Analyzer

	return goanalysis.NewLinter(a.Name, a.Doc, []*analysis.Analyzer{a}, nil).
		WithLoadMode(goanalysis.LoadModeTypesInfo)
}
