package golinters

import (
	"fmt"
	"sync"

	goconstAPI "github.com/jgautheron/goconst"
	"golang.org/x/tools/go/analysis"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const goconstName = "goconst"

func NewGoconst() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: goconstName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		goconstName,
		"Finds repeated strings that could be replaced by a constant",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			issues, err := checkConstants(pass, lintCtx)
			if err != nil || len(issues) == 0 {
				return nil, err
			}

			mu.Lock()
			resIssues = append(resIssues, issues...)
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeSyntax)
}

func checkConstants(pass *analysis.Pass, lintCtx *linter.Context) ([]goanalysis.Issue, error) {
	settings := lintCtx.Settings().Goconst

	cfg := goconstAPI.Config{
		IgnoreTests:        settings.IgnoreTests,
		MatchWithConstants: settings.MatchWithConstants,
		MinStringLength:    settings.MinStringLen,
		MinOccurrences:     settings.MinOccurrencesCount,
		ParseNumbers:       settings.ParseNumbers,
		NumberMin:          settings.NumberMin,
		NumberMax:          settings.NumberMax,
		ExcludeTypes:       map[goconstAPI.Type]bool{},
	}

	if settings.IgnoreCalls {
		cfg.ExcludeTypes[goconstAPI.Call] = true
	}

	goconstIssues, err := goconstAPI.Run(pass.Files, pass.Fset, &cfg)
	if err != nil {
		return nil, err
	}

	if len(goconstIssues) == 0 {
		return nil, nil
	}

	res := make([]goanalysis.Issue, 0, len(goconstIssues))
	for _, i := range goconstIssues {
		textBegin := fmt.Sprintf("string %s has %d occurrences", formatCode(i.Str, lintCtx.Cfg), i.OccurrencesCount)
		var textEnd string
		if i.MatchingConst == "" {
			textEnd = ", make it a constant"
		} else {
			textEnd = fmt.Sprintf(", but such constant %s already exists", formatCode(i.MatchingConst, lintCtx.Cfg))
		}
		res = append(res, goanalysis.NewIssue(&result.Issue{
			Pos:        i.Pos,
			Text:       textBegin + textEnd,
			FromLinter: goconstName,
		}, pass))
	}

	return res, nil
}
