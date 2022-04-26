package golinters

import (
	"fmt"
	"go/ast"
	"sync"

	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/golinters/nolintlint"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const NolintlintName = "nolintlint"

func NewNoLintLint() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: NolintlintName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		NolintlintName,
		"Reports ill-formed or insufficient nolint directives",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			var needs nolintlint.Needs
			settings := lintCtx.Settings().NoLintLint
			if settings.RequireExplanation {
				needs |= nolintlint.NeedsExplanation
			}
			if !settings.AllowLeadingSpace {
				needs |= nolintlint.NeedsMachineOnly
			}
			if settings.RequireSpecific {
				needs |= nolintlint.NeedsSpecific
			}
			if !settings.AllowUnused {
				needs |= nolintlint.NeedsUnused
			}

			lnt, err := nolintlint.NewLinter(needs, settings.AllowNoExplanation)
			if err != nil {
				return nil, err
			}

			nodes := make([]ast.Node, 0, len(pass.Files))
			for _, n := range pass.Files {
				nodes = append(nodes, n)
			}
			issues, err := lnt.Run(pass.Fset, nodes...)
			if err != nil {
				return nil, fmt.Errorf("linter failed to run: %s", err)
			}
			var res []goanalysis.Issue
			for _, i := range issues {
				expectNoLint := false
				var expectedNolintLinter string
				if ii, ok := i.(nolintlint.UnusedCandidate); ok {
					expectedNolintLinter = ii.ExpectedLinter
					expectNoLint = true
				}
				issue := &result.Issue{
					FromLinter:           NolintlintName,
					Text:                 i.Details(),
					Pos:                  i.Position(),
					ExpectNoLint:         expectNoLint,
					ExpectedNoLintLinter: expectedNolintLinter,
					Replacement:          i.Replacement(),
				}
				res = append(res, goanalysis.NewIssue(issue, pass))
			}

			if len(res) == 0 {
				return nil, nil
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
