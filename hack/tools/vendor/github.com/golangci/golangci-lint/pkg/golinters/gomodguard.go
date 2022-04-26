package golinters

import (
	"sync"

	"github.com/ryancurrah/gomodguard"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const (
	gomodguardName = "gomodguard"
	gomodguardDesc = "Allow and block list linter for direct Go module dependencies. " +
		"This is different from depguard where there are different block " +
		"types for example version constraints and module recommendations."
)

// NewGomodguard returns a new Gomodguard linter.
func NewGomodguard() *goanalysis.Linter {
	var (
		issues   []goanalysis.Issue
		mu       = sync.Mutex{}
		analyzer = &analysis.Analyzer{
			Name: goanalysis.TheOnlyAnalyzerName,
			Doc:  goanalysis.TheOnlyanalyzerDoc,
		}
	)

	return goanalysis.NewLinter(
		gomodguardName,
		gomodguardDesc,
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		linterCfg := lintCtx.Cfg.LintersSettings.Gomodguard

		processorCfg := &gomodguard.Configuration{}
		processorCfg.Allowed.Modules = linterCfg.Allowed.Modules
		processorCfg.Allowed.Domains = linterCfg.Allowed.Domains
		processorCfg.Blocked.LocalReplaceDirectives = linterCfg.Blocked.LocalReplaceDirectives

		for n := range linterCfg.Blocked.Modules {
			for k, v := range linterCfg.Blocked.Modules[n] {
				m := map[string]gomodguard.BlockedModule{k: {
					Recommendations: v.Recommendations,
					Reason:          v.Reason,
				}}
				processorCfg.Blocked.Modules = append(processorCfg.Blocked.Modules, m)
				break
			}
		}

		for n := range linterCfg.Blocked.Versions {
			for k, v := range linterCfg.Blocked.Versions[n] {
				m := map[string]gomodguard.BlockedVersion{k: {
					Version: v.Version,
					Reason:  v.Reason,
				}}
				processorCfg.Blocked.Versions = append(processorCfg.Blocked.Versions, m)
				break
			}
		}

		processor, err := gomodguard.NewProcessor(processorCfg)
		if err != nil {
			lintCtx.Log.Warnf("running gomodguard failed: %s: if you are not using go modules "+
				"it is suggested to disable this linter", err)
			return
		}

		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var files []string

			for _, file := range pass.Files {
				files = append(files, pass.Fset.PositionFor(file.Pos(), false).Filename)
			}

			gomodguardIssues := processor.ProcessFiles(files)

			mu.Lock()
			defer mu.Unlock()

			for _, gomodguardIssue := range gomodguardIssues {
				issues = append(issues, goanalysis.NewIssue(&result.Issue{
					FromLinter: gomodguardName,
					Pos:        gomodguardIssue.Position,
					Text:       gomodguardIssue.Reason,
				}, pass))
			}

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return issues
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}
