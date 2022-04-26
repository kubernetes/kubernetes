package golinters

import (
	"strings"

	"github.com/breml/bidichk/pkg/bidichk"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewBiDiChkFuncName(cfg *config.BiDiChkSettings) *goanalysis.Linter {
	a := bidichk.NewAnalyzer()

	cfgMap := map[string]map[string]interface{}{}
	if cfg != nil {
		var opts []string

		if cfg.LeftToRightEmbedding {
			opts = append(opts, "LEFT-TO-RIGHT-EMBEDDING")
		}
		if cfg.RightToLeftEmbedding {
			opts = append(opts, "RIGHT-TO-LEFT-EMBEDDING")
		}
		if cfg.PopDirectionalFormatting {
			opts = append(opts, "POP-DIRECTIONAL-FORMATTING")
		}
		if cfg.LeftToRightOverride {
			opts = append(opts, "LEFT-TO-RIGHT-OVERRIDE")
		}
		if cfg.RightToLeftOverride {
			opts = append(opts, "RIGHT-TO-LEFT-OVERRIDE")
		}
		if cfg.LeftToRightIsolate {
			opts = append(opts, "LEFT-TO-RIGHT-ISOLATE")
		}
		if cfg.RightToLeftIsolate {
			opts = append(opts, "RIGHT-TO-LEFT-ISOLATE")
		}
		if cfg.FirstStrongIsolate {
			opts = append(opts, "FIRST-STRONG-ISOLATE")
		}
		if cfg.PopDirectionalIsolate {
			opts = append(opts, "POP-DIRECTIONAL-ISOLATE")
		}

		cfgMap[a.Name] = map[string]interface{}{
			"disallowed-runes": strings.Join(opts, ","),
		}
	}

	return goanalysis.NewLinter(
		"bidichk",
		"Checks for dangerous unicode character sequences",
		[]*analysis.Analyzer{a},
		cfgMap,
	).WithLoadMode(goanalysis.LoadModeSyntax)
}
