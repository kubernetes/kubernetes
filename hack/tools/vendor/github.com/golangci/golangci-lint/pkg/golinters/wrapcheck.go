package golinters

import (
	"github.com/tomarrell/wrapcheck/v2/wrapcheck"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

const wrapcheckName = "wrapcheck"

func NewWrapcheck(settings *config.WrapcheckSettings) *goanalysis.Linter {
	cfg := wrapcheck.NewDefaultConfig()
	if settings != nil {
		if len(settings.IgnoreSigs) != 0 {
			cfg.IgnoreSigs = settings.IgnoreSigs
		}
		if len(settings.IgnoreSigRegexps) != 0 {
			cfg.IgnoreSigRegexps = settings.IgnoreSigRegexps
		}
		if len(settings.IgnorePackageGlobs) != 0 {
			cfg.IgnorePackageGlobs = settings.IgnorePackageGlobs
		}
	}

	a := wrapcheck.NewAnalyzer(cfg)

	return goanalysis.NewLinter(
		wrapcheckName,
		a.Doc,
		[]*analysis.Analyzer{a},
		nil,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
