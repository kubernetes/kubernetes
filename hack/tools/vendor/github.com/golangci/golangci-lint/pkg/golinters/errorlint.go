package golinters

import (
	"github.com/polyfloyd/go-errorlint/errorlint"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewErrorLint(cfg *config.ErrorLintSettings) *goanalysis.Linter {
	a := errorlint.NewAnalyzer()

	cfgMap := map[string]map[string]interface{}{}

	if cfg != nil {
		cfgMap[a.Name] = map[string]interface{}{
			"errorf":     cfg.Errorf,
			"asserts":    cfg.Asserts,
			"comparison": cfg.Comparison,
		}
	}

	return goanalysis.NewLinter(
		a.Name,
		"errorlint is a linter for that can be used to find code "+
			"that will cause problems with the error wrapping scheme introduced in Go 1.13.",
		[]*analysis.Analyzer{a},
		cfgMap,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
