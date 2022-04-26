package golinters

import (
	"github.com/bkielbasa/cyclop/pkg/analyzer"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

const cyclopName = "cyclop"

func NewCyclop(settings *config.Cyclop) *goanalysis.Linter {
	a := analyzer.NewAnalyzer()

	var cfg map[string]map[string]interface{}
	if settings != nil {
		d := map[string]interface{}{
			"skipTests": settings.SkipTests,
		}

		if settings.MaxComplexity != 0 {
			d["maxComplexity"] = settings.MaxComplexity
		}

		if settings.PackageAverage != 0 {
			d["packageAverage"] = settings.PackageAverage
		}

		cfg = map[string]map[string]interface{}{a.Name: d}
	}

	return goanalysis.NewLinter(
		cyclopName,
		"checks function and package cyclomatic complexity",
		[]*analysis.Analyzer{a},
		cfg,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
