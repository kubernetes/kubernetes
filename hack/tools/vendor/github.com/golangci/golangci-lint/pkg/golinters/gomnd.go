package golinters

import (
	mnd "github.com/tommy-muehle/go-mnd/v2"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewGoMND(settings *config.GoMndSettings) *goanalysis.Linter {
	var linterCfg map[string]map[string]interface{}

	if settings != nil {
		// TODO(ldez) For compatibility only, must be drop in v2.
		if len(settings.Settings) > 0 {
			linterCfg = settings.Settings
		} else {
			cfg := make(map[string]interface{})
			if len(settings.Checks) > 0 {
				cfg["checks"] = settings.Checks
			}
			if len(settings.IgnoredNumbers) > 0 {
				cfg["ignored-numbers"] = settings.IgnoredNumbers
			}
			if len(settings.IgnoredFiles) > 0 {
				cfg["ignored-files"] = settings.IgnoredFiles
			}
			if len(settings.IgnoredFunctions) > 0 {
				cfg["ignored-functions"] = settings.IgnoredFunctions
			}

			linterCfg = map[string]map[string]interface{}{
				"mnd": cfg,
			}
		}
	}

	return goanalysis.NewLinter(
		"gomnd",
		"An analyzer to detect magic numbers.",
		[]*analysis.Analyzer{mnd.Analyzer},
		linterCfg,
	).WithLoadMode(goanalysis.LoadModeSyntax)
}
