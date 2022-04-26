package golinters // nolint:dupl

import (
	"fmt"
	"sync"

	varcheckAPI "github.com/golangci/check/cmd/varcheck"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

func NewVarcheck() *goanalysis.Linter {
	const linterName = "varcheck"
	var mu sync.Mutex
	var res []goanalysis.Issue
	analyzer := &analysis.Analyzer{
		Name: linterName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		linterName,
		"Finds unused global variables and constants",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		checkExported := lintCtx.Settings().Varcheck.CheckExportedFields
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			prog := goanalysis.MakeFakeLoaderProgram(pass)

			varcheckIssues := varcheckAPI.Run(prog, checkExported)
			if len(varcheckIssues) == 0 {
				return nil, nil
			}

			issues := make([]goanalysis.Issue, 0, len(varcheckIssues))
			for _, i := range varcheckIssues {
				issues = append(issues, goanalysis.NewIssue(&result.Issue{
					Pos:        i.Pos,
					Text:       fmt.Sprintf("%s is unused", formatCode(i.VarName, lintCtx.Cfg)),
					FromLinter: linterName,
				}, pass))
			}

			mu.Lock()
			res = append(res, issues...)
			mu.Unlock()
			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return res
	}).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
