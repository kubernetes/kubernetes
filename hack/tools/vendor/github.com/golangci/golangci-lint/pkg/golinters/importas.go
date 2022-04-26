package golinters

import (
	"fmt"
	"strconv"

	"github.com/julz/importas" // nolint: misspell
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
)

func NewImportAs(settings *config.ImportAsSettings) *goanalysis.Linter {
	analyzer := importas.Analyzer

	return goanalysis.NewLinter(
		analyzer.Name,
		analyzer.Doc,
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		if settings == nil {
			return
		}
		if len(settings.Alias) == 0 {
			lintCtx.Log.Infof("importas settings found, but no aliases listed. List aliases under alias: key.") // nolint: misspell
		}

		if err := analyzer.Flags.Set("no-unaliased", strconv.FormatBool(settings.NoUnaliased)); err != nil {
			lintCtx.Log.Errorf("failed to parse configuration: %v", err)
		}

		if err := analyzer.Flags.Set("no-extra-aliases", strconv.FormatBool(settings.NoExtraAliases)); err != nil {
			lintCtx.Log.Errorf("failed to parse configuration: %v", err)
		}

		for _, a := range settings.Alias {
			if a.Pkg == "" {
				lintCtx.Log.Errorf("invalid configuration, empty package: pkg=%s alias=%s", a.Pkg, a.Alias)
				continue
			}

			err := analyzer.Flags.Set("alias", fmt.Sprintf("%s:%s", a.Pkg, a.Alias))
			if err != nil {
				lintCtx.Log.Errorf("failed to parse configuration: %v", err)
			}
		}
	}).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
