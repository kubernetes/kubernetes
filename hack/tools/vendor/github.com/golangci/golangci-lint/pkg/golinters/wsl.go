package golinters

import (
	"sync"

	"github.com/bombsimon/wsl/v3"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const (
	name = "wsl"
)

// NewWSL returns a new WSL linter.
func NewWSL() *goanalysis.Linter {
	var (
		issues   []goanalysis.Issue
		mu       = sync.Mutex{}
		analyzer = &analysis.Analyzer{
			Name: goanalysis.TheOnlyAnalyzerName,
			Doc:  goanalysis.TheOnlyanalyzerDoc,
		}
	)

	return goanalysis.NewLinter(
		name,
		"Whitespace Linter - Forces you to use empty lines!",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var (
				files        = make([]string, 0, len(pass.Files))
				linterCfg    = lintCtx.Cfg.LintersSettings.WSL
				processorCfg = wsl.Configuration{
					StrictAppend:                     linterCfg.StrictAppend,
					AllowAssignAndCallCuddle:         linterCfg.AllowAssignAndCallCuddle,
					AllowAssignAndAnythingCuddle:     linterCfg.AllowAssignAndAnythingCuddle,
					AllowMultiLineAssignCuddle:       linterCfg.AllowMultiLineAssignCuddle,
					AllowCuddleDeclaration:           linterCfg.AllowCuddleDeclaration,
					AllowTrailingComment:             linterCfg.AllowTrailingComment,
					AllowSeparatedLeadingComment:     linterCfg.AllowSeparatedLeadingComment,
					ForceCuddleErrCheckAndAssign:     linterCfg.ForceCuddleErrCheckAndAssign,
					ForceCaseTrailingWhitespaceLimit: linterCfg.ForceCaseTrailingWhitespaceLimit,
					ForceExclusiveShortDeclarations:  linterCfg.ForceExclusiveShortDeclarations,
					AllowCuddleWithCalls:             []string{"Lock", "RLock"},
					AllowCuddleWithRHS:               []string{"Unlock", "RUnlock"},
					ErrorVariableNames:               []string{"err"},
				}
			)

			for _, file := range pass.Files {
				files = append(files, pass.Fset.PositionFor(file.Pos(), false).Filename)
			}

			wslErrors, _ := wsl.NewProcessorWithConfig(processorCfg).
				ProcessFiles(files)

			if len(wslErrors) == 0 {
				return nil, nil
			}

			mu.Lock()
			defer mu.Unlock()

			for _, err := range wslErrors {
				issues = append(issues, goanalysis.NewIssue(&result.Issue{
					FromLinter: name,
					Pos:        err.Position,
					Text:       err.Reason,
				}, pass))
			}

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return issues
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}
