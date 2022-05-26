package golinters

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sync"

	lintAPI "github.com/golangci/lint-1"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

func golintProcessPkg(minConfidence float64, files []*ast.File, fset *token.FileSet,
	typesPkg *types.Package, typesInfo *types.Info) ([]result.Issue, error) {
	l := new(lintAPI.Linter)
	ps, err := l.LintPkg(files, fset, typesPkg, typesInfo)
	if err != nil {
		return nil, fmt.Errorf("can't lint %d files: %s", len(files), err)
	}

	if len(ps) == 0 {
		return nil, nil
	}

	issues := make([]result.Issue, 0, len(ps)) // This is worst case
	for idx := range ps {
		if ps[idx].Confidence >= minConfidence {
			issues = append(issues, result.Issue{
				Pos:        ps[idx].Position,
				Text:       ps[idx].Text,
				FromLinter: golintName,
			})
			// TODO: use p.Link and p.Category
		}
	}

	return issues, nil
}

const golintName = "golint"

func NewGolint() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: golintName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		golintName,
		"Golint differs from gofmt. Gofmt reformats Go source code, whereas golint prints out style mistakes",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			res, err := golintProcessPkg(lintCtx.Settings().Golint.MinConfidence, pass.Files, pass.Fset, pass.Pkg, pass.TypesInfo)
			if err != nil || len(res) == 0 {
				return nil, err
			}

			mu.Lock()
			for i := range res {
				resIssues = append(resIssues, goanalysis.NewIssue(&res[i], pass))
			}
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeTypesInfo)
}
