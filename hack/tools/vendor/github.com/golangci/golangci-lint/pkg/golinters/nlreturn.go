package golinters

import (
	"github.com/ssgreg/nlreturn/v2/pkg/nlreturn"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewNLReturn(settings *config.NlreturnSettings) *goanalysis.Linter {
	a := nlreturn.NewAnalyzer()

	cfg := map[string]map[string]interface{}{}
	if settings != nil {
		cfg[a.Name] = map[string]interface{}{
			"block-size": settings.BlockSize,
		}
	}

	return goanalysis.NewLinter(
		a.Name,
		"nlreturn checks for a new line before return and branch statements to increase code clarity",
		[]*analysis.Analyzer{a},
		cfg,
	).WithLoadMode(goanalysis.LoadModeSyntax)
}
