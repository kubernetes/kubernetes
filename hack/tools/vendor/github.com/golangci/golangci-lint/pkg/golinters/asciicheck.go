package golinters

import (
	"github.com/tdakkota/asciicheck"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewAsciicheck() *goanalysis.Linter {
	return goanalysis.NewLinter(
		"asciicheck",
		"Simple linter to check that your code does not contain non-ASCII identifiers",
		[]*analysis.Analyzer{
			asciicheck.NewAnalyzer(),
		},
		nil,
	).WithLoadMode(goanalysis.LoadModeSyntax)
}
