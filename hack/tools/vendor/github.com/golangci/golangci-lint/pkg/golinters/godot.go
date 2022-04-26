package golinters

import (
	"sync"

	"github.com/tetafro/godot"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const godotName = "godot"

func NewGodot() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: godotName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		godotName,
		"Check if comments end in a period",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		cfg := lintCtx.Cfg.LintersSettings.Godot
		settings := godot.Settings{
			Scope:   godot.Scope(cfg.Scope),
			Exclude: cfg.Exclude,
			Period:  cfg.Period,
			Capital: cfg.Capital,
		}

		// Convert deprecated setting
		// todo(butuzov): remove on v2 release
		if cfg.CheckAll { // nolint:staticcheck
			settings.Scope = godot.AllScope
		}

		if settings.Scope == "" {
			settings.Scope = godot.DeclScope
		}

		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var issues []godot.Issue
			for _, file := range pass.Files {
				iss, err := godot.Run(file, pass.Fset, settings)
				if err != nil {
					return nil, err
				}
				issues = append(issues, iss...)
			}

			if len(issues) == 0 {
				return nil, nil
			}

			res := make([]goanalysis.Issue, len(issues))
			for k, i := range issues {
				issue := result.Issue{
					Pos:        i.Pos,
					Text:       i.Message,
					FromLinter: godotName,
					Replacement: &result.Replacement{
						NewLines: []string{i.Replacement},
					},
				}

				res[k] = goanalysis.NewIssue(&issue, pass)
			}

			mu.Lock()
			resIssues = append(resIssues, res...)
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}
