package golinters

import (
	"github.com/Djarvur/go-err113"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewGoerr113() *goanalysis.Linter {
	return goanalysis.NewLinter(
		"goerr113",
		"Golang linter to check the errors handling expressions",
		[]*analysis.Analyzer{
			err113.NewAnalyzer(),
		},
		nil,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
