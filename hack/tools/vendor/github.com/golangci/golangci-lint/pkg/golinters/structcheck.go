package golinters // nolint:dupl

import (
	"fmt"
	"sync"

	structcheckAPI "github.com/golangci/check/cmd/structcheck"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

func NewStructcheck() *goanalysis.Linter {
	const linterName = "structcheck"
	var mu sync.Mutex
	var res []goanalysis.Issue
	analyzer := &analysis.Analyzer{
		Name: linterName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		linterName,
		"Finds unused struct fields",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		checkExported := lintCtx.Settings().Structcheck.CheckExportedFields
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			prog := goanalysis.MakeFakeLoaderProgram(pass)

			structcheckIssues := structcheckAPI.Run(prog, checkExported)
			if len(structcheckIssues) == 0 {
				return nil, nil
			}

			issues := make([]goanalysis.Issue, 0, len(structcheckIssues))
			for _, i := range structcheckIssues {
				issues = append(issues, goanalysis.NewIssue(&result.Issue{
					Pos:        i.Pos,
					Text:       fmt.Sprintf("%s is unused", formatCode(i.FieldName, lintCtx.Cfg)),
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
