package golinters

import (
	"github.com/esimonov/ifshort/pkg/analyzer"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewIfshort(settings *config.IfshortSettings) *goanalysis.Linter {
	var cfg map[string]map[string]interface{}
	if settings != nil {
		cfg = map[string]map[string]interface{}{
			analyzer.Analyzer.Name: {
				"max-decl-lines": settings.MaxDeclLines,
				"max-decl-chars": settings.MaxDeclChars,
			},
		}
	}

	return goanalysis.NewLinter(
		"ifshort",
		"Checks that your code uses short syntax for if-statements whenever possible",
		[]*analysis.Analyzer{analyzer.Analyzer},
		cfg,
	).WithLoadMode(goanalysis.LoadModeSyntax)
}
