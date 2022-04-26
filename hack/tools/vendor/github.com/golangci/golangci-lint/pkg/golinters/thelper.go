package golinters

import (
	"strings"

	"github.com/kulti/thelper/pkg/analyzer"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewThelper(cfg *config.ThelperSettings) *goanalysis.Linter {
	a := analyzer.NewAnalyzer()

	cfgMap := map[string]map[string]interface{}{}
	if cfg != nil {
		var opts []string

		if cfg.Test.Name {
			opts = append(opts, "t_name")
		}
		if cfg.Test.Begin {
			opts = append(opts, "t_begin")
		}
		if cfg.Test.First {
			opts = append(opts, "t_first")
		}

		if cfg.Benchmark.Name {
			opts = append(opts, "b_name")
		}
		if cfg.Benchmark.Begin {
			opts = append(opts, "b_begin")
		}
		if cfg.Benchmark.First {
			opts = append(opts, "b_first")
		}

		if cfg.TB.Name {
			opts = append(opts, "tb_name")
		}
		if cfg.TB.Begin {
			opts = append(opts, "tb_begin")
		}
		if cfg.TB.First {
			opts = append(opts, "tb_first")
		}

		cfgMap[a.Name] = map[string]interface{}{
			"checks": strings.Join(opts, ","),
		}
	}

	return goanalysis.NewLinter(
		"thelper",
		"thelper detects golang test helpers without t.Helper() call and checks the consistency of test helpers",
		[]*analysis.Analyzer{a},
		cfgMap,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
