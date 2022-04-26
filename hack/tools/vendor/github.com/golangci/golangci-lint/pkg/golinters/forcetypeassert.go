package golinters

import (
	"github.com/gostaticanalysis/forcetypeassert"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewForceTypeAssert() *goanalysis.Linter {
	a := forcetypeassert.Analyzer

	return goanalysis.NewLinter(
		a.Name,
		"finds forced type assertions",
		[]*analysis.Analyzer{a},
		nil,
	).WithLoadMode(goanalysis.LoadModeSyntax)
}
