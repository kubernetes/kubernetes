package golinters

import (
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/buildssa"
	"mvdan.cc/interfacer/check"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const interfacerName = "interfacer"

func NewInterfacer() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name:     interfacerName,
		Doc:      goanalysis.TheOnlyanalyzerDoc,
		Requires: []*analysis.Analyzer{buildssa.Analyzer},
	}
	return goanalysis.NewLinter(
		interfacerName,
		"Linter that suggests narrower interface types",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			ssa := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA)
			ssaPkg := ssa.Pkg
			c := &check.Checker{}
			prog := goanalysis.MakeFakeLoaderProgram(pass)
			c.Program(prog)
			c.ProgramSSA(ssaPkg.Prog)

			issues, err := c.Check()
			if err != nil {
				return nil, err
			}
			if len(issues) == 0 {
				return nil, nil
			}

			res := make([]goanalysis.Issue, 0, len(issues))
			for _, i := range issues {
				pos := pass.Fset.Position(i.Pos())
				res = append(res, goanalysis.NewIssue(&result.Issue{
					Pos:        pos,
					Text:       i.Message(),
					FromLinter: interfacerName,
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
