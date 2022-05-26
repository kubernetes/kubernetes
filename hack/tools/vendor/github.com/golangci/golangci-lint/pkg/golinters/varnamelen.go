package golinters

import (
	"strconv"
	"strings"

	"github.com/blizzy78/varnamelen"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewVarnamelen(settings *config.VarnamelenSettings) *goanalysis.Linter {
	a := varnamelen.NewAnalyzer()

	cfg := map[string]map[string]interface{}{}
	if settings != nil {
		vnlCfg := map[string]interface{}{
			"checkReceiver":      strconv.FormatBool(settings.CheckReceiver),
			"checkReturn":        strconv.FormatBool(settings.CheckReturn),
			"ignoreNames":        strings.Join(settings.IgnoreNames, ","),
			"ignoreTypeAssertOk": strconv.FormatBool(settings.IgnoreTypeAssertOk),
			"ignoreMapIndexOk":   strconv.FormatBool(settings.IgnoreMapIndexOk),
			"ignoreChanRecvOk":   strconv.FormatBool(settings.IgnoreChanRecvOk),
			"ignoreDecls":        strings.Join(settings.IgnoreDecls, ","),
		}

		if settings.MaxDistance > 0 {
			vnlCfg["maxDistance"] = strconv.Itoa(settings.MaxDistance)
		}
		if settings.MinNameLength > 0 {
			vnlCfg["minNameLength"] = strconv.Itoa(settings.MinNameLength)
		}

		cfg[a.Name] = vnlCfg
	}

	return goanalysis.NewLinter(
		a.Name,
		"checks that the length of a variable's name matches its scope",
		[]*analysis.Analyzer{a},
		cfg,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
