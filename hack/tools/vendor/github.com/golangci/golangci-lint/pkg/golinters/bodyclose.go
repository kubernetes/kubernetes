package golinters

import (
	"github.com/timakin/bodyclose/passes/bodyclose"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewBodyclose() *goanalysis.Linter {
	analyzers := []*analysis.Analyzer{
		bodyclose.Analyzer,
	}

	return goanalysis.NewLinter(
		"bodyclose",
		"checks whether HTTP response body is closed successfully",
		analyzers,
		nil,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
