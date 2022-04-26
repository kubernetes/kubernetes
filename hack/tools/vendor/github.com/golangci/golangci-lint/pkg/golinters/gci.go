package golinters

import (
	"fmt"
	"strings"

	gci "github.com/daixiang0/gci/pkg/analyzer"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
)

const gciName = "gci"

func NewGci(settings *config.GciSettings) *goanalysis.Linter {
	var linterCfg map[string]map[string]interface{}

	if settings != nil {
		cfg := map[string]interface{}{
			gci.NoInlineCommentsFlag:  settings.NoInlineComments,
			gci.NoPrefixCommentsFlag:  settings.NoPrefixComments,
			gci.SectionsFlag:          strings.Join(settings.Sections, gci.SectionDelimiter),
			gci.SectionSeparatorsFlag: strings.Join(settings.SectionSeparator, gci.SectionDelimiter),
		}

		if settings.LocalPrefixes != "" {
			prefix := []string{"standard", "default", fmt.Sprintf("prefix(%s)", settings.LocalPrefixes)}
			cfg[gci.SectionsFlag] = strings.Join(prefix, gci.SectionDelimiter)
		}

		linterCfg = map[string]map[string]interface{}{
			gci.Analyzer.Name: cfg,
		}
	}

	return goanalysis.NewLinter(
		gciName,
		"Gci controls golang package import order and makes it always deterministic.",
		[]*analysis.Analyzer{gci.Analyzer},
		linterCfg,
	).WithContextSetter(func(lintCtx *linter.Context) {
		if settings.LocalPrefixes != "" {
			lintCtx.Log.Warnf("gci: `local-prefixes` is deprecated, use `sections` and `prefix(%s)` instead.", settings.LocalPrefixes)
		}
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}
