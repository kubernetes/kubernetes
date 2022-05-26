package golinters

import (
	"github.com/ldez/tagliatelle"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewTagliatelle(settings *config.TagliatelleSettings) *goanalysis.Linter {
	cfg := tagliatelle.Config{
		Rules: map[string]string{
			"json": "camel",
			"yaml": "camel",
		},
	}

	if settings != nil {
		for k, v := range settings.Case.Rules {
			cfg.Rules[k] = v
		}
		cfg.UseFieldName = settings.Case.UseFieldName
	}

	a := tagliatelle.New(cfg)

	return goanalysis.NewLinter(a.Name, a.Doc, []*analysis.Analyzer{a}, nil).
		WithLoadMode(goanalysis.LoadModeSyntax)
}
