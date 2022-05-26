package golinters

import (
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"golang.org/x/tools/go/packages"
	"mvdan.cc/unparam/check"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

func NewUnparam() *goanalysis.Linter {
	const linterName = "unparam"
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name:     linterName,
		Doc:      goanalysis.TheOnlyanalyzerDoc,
		Requires: []*analysis.Analyzer{buildssa.Analyzer},
	}
	return goanalysis.NewLinter(
		linterName,
		"Reports unused function parameters",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		us := &lintCtx.Settings().Unparam
		if us.Algo != "cha" {
			lintCtx.Log.Warnf("`linters-settings.unparam.algo` isn't supported by the newest `unparam`")
		}

		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			ssa := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA)
			ssaPkg := ssa.Pkg

			pkg := &packages.Package{
				Fset:      pass.Fset,
				Syntax:    pass.Files,
				Types:     pass.Pkg,
				TypesInfo: pass.TypesInfo,
			}

			c := &check.Checker{}
			c.CheckExportedFuncs(us.CheckExported)
			c.Packages([]*packages.Package{pkg})
			c.ProgramSSA(ssaPkg.Prog)

			unparamIssues, err := c.Check()
			if err != nil {
				return nil, err
			}

			var res []goanalysis.Issue
			for _, i := range unparamIssues {
				res = append(res, goanalysis.NewIssue(&result.Issue{
					Pos:        pass.Fset.Position(i.Pos()),
					Text:       i.Message(),
					FromLinter: linterName,
				}, pass))
			}

			mu.Lock()
			resIssues = append(resIssues, res...)
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
