package golinters

import (
	grouper "github.com/leonklingele/grouper/pkg/analyzer"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewGrouper(settings *config.GrouperSettings) *goanalysis.Linter {
	linterCfg := map[string]map[string]interface{}{}
	if settings != nil {
		linterCfg["grouper"] = map[string]interface{}{
			"const-require-single-const":   settings.ConstRequireSingleConst,
			"const-require-grouping":       settings.ConstRequireGrouping,
			"import-require-single-import": settings.ImportRequireSingleImport,
			"import-require-grouping":      settings.ImportRequireGrouping,
			"type-require-single-type":     settings.TypeRequireSingleType,
			"type-require-grouping":        settings.TypeRequireGrouping,
			"var-require-single-var":       settings.VarRequireSingleVar,
			"var-require-grouping":         settings.VarRequireGrouping,
		}
	}

	return goanalysis.NewLinter(
		"grouper",
		"An analyzer to analyze expression groups.",
		[]*analysis.Analyzer{grouper.New()},
		linterCfg,
	).WithLoadMode(goanalysis.LoadModeSyntax)
}
