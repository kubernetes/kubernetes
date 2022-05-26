package golinters

import (
	"strings"

	"gitlab.com/bosi/decorder"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
)

func NewDecorder(settings *config.DecorderSettings) *goanalysis.Linter {
	a := decorder.Analyzer

	analyzers := []*analysis.Analyzer{a}

	// disable all rules/checks by default
	cfg := map[string]interface{}{
		"disable-dec-num-check":         true,
		"disable-dec-order-check":       true,
		"disable-init-func-first-check": true,
	}

	if settings != nil {
		cfg["dec-order"] = strings.Join(settings.DecOrder, ",")
		cfg["disable-dec-num-check"] = settings.DisableDecNumCheck
		cfg["disable-dec-order-check"] = settings.DisableDecOrderCheck
		cfg["disable-init-func-first-check"] = settings.DisableInitFuncFirstCheck
	}

	return goanalysis.NewLinter(
		a.Name,
		a.Doc,
		analyzers,
		map[string]map[string]interface{}{a.Name: cfg},
	).WithLoadMode(goanalysis.LoadModeSyntax)
}
