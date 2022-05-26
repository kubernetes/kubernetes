package golinters

import (
	"github.com/breml/errchkjson"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewErrChkJSONFuncName(cfg *config.ErrChkJSONSettings) *goanalysis.Linter {
	a := errchkjson.NewAnalyzer()

	cfgMap := map[string]map[string]interface{}{}
	cfgMap[a.Name] = map[string]interface{}{
		"omit-safe": true,
	}
	if cfg != nil {
		cfgMap[a.Name] = map[string]interface{}{
			"omit-safe":          !cfg.CheckErrorFreeEncoding,
			"report-no-exported": cfg.ReportNoExported,
		}
	}

	return goanalysis.NewLinter(
		"errchkjson",
		"Checks types passed to the json encoding functions. "+
			"Reports unsupported types and optionally reports occations, "+
			"where the check for the returned error can be omitted.",
		[]*analysis.Analyzer{a},
		cfgMap,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
