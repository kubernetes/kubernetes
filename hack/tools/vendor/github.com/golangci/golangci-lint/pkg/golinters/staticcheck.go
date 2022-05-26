package golinters

import (
	"honnef.co/go/tools/staticcheck"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewStaticcheck(settings *config.StaticCheckSettings) *goanalysis.Linter {
	cfg := staticCheckConfig(settings)

	analyzers := setupStaticCheckAnalyzers(staticcheck.Analyzers, getGoVersion(settings), cfg.Checks)

	return goanalysis.NewLinter(
		"staticcheck",
		"Staticcheck is a go vet on steroids, applying a ton of static analysis checks",
		analyzers,
		nil,
	).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
