package golinters

import (
	"github.com/maratori/testpackage/pkg/testpackage"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewTestpackage(cfg *config.TestpackageSettings) *goanalysis.Linter {
	var a = testpackage.NewAnalyzer()
	var settings map[string]map[string]interface{}
	if cfg != nil {
		settings = map[string]map[string]interface{}{
			a.Name: {
				testpackage.SkipRegexpFlagName: cfg.SkipRegexp,
			},
		}
	}
	return goanalysis.NewLinter(a.Name, a.Doc, []*analysis.Analyzer{a}, settings).
		WithLoadMode(goanalysis.LoadModeSyntax)
}
