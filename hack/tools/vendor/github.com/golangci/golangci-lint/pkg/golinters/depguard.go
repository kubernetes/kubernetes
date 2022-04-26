package golinters

import (
	"fmt"
	"strings"
	"sync"

	"github.com/OpenPeeDeeP/depguard"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/loader" //nolint:staticcheck // require changes in github.com/OpenPeeDeeP/depguard

	"github.com/golangci/golangci-lint/pkg/config"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
)

const depguardLinterName = "depguard"

func NewDepguard() *goanalysis.Linter {
	var mu sync.Mutex
	var resIssues []goanalysis.Issue

	analyzer := &analysis.Analyzer{
		Name: depguardLinterName,
		Doc:  goanalysis.TheOnlyanalyzerDoc,
	}
	return goanalysis.NewLinter(
		depguardLinterName,
		"Go linter that checks if package imports are in a list of acceptable packages",
		[]*analysis.Analyzer{analyzer},
		nil,
	).WithContextSetter(func(lintCtx *linter.Context) {
		dg, err := newDepGuard(&lintCtx.Settings().Depguard)

		analyzer.Run = func(pass *analysis.Pass) (interface{}, error) {
			if err != nil {
				return nil, err
			}

			issues, errRun := dg.run(pass)
			if errRun != nil {
				return nil, errRun
			}

			mu.Lock()
			resIssues = append(resIssues, issues...)
			mu.Unlock()

			return nil, nil
		}
	}).WithIssuesReporter(func(*linter.Context) []goanalysis.Issue {
		return resIssues
	}).WithLoadMode(goanalysis.LoadModeTypesInfo)
}

type depGuard struct {
	loadConfig *loader.Config
	guardians  []*guardian
}

func newDepGuard(settings *config.DepGuardSettings) (*depGuard, error) {
	ps, err := newGuardian(settings)
	if err != nil {
		return nil, err
	}

	d := &depGuard{
		loadConfig: &loader.Config{
			Cwd:   "",  // fallbacked to os.Getcwd
			Build: nil, // fallbacked to build.Default
		},
		guardians: []*guardian{ps},
	}

	for _, additional := range settings.AdditionalGuards {
		add := additional
		ps, err = newGuardian(&add)
		if err != nil {
			return nil, err
		}

		d.guardians = append(d.guardians, ps)
	}

	return d, nil
}

func (d depGuard) run(pass *analysis.Pass) ([]goanalysis.Issue, error) {
	prog := goanalysis.MakeFakeLoaderProgram(pass)

	var resIssues []goanalysis.Issue
	for _, g := range d.guardians {
		issues, errRun := g.run(d.loadConfig, prog, pass)
		if errRun != nil {
			return nil, errRun
		}

		resIssues = append(resIssues, issues...)
	}

	return resIssues, nil
}

type guardian struct {
	*depguard.Depguard
	pkgsWithErrorMessage map[string]string
}

func newGuardian(settings *config.DepGuardSettings) (*guardian, error) {
	dg := &depguard.Depguard{
		Packages:        settings.Packages,
		IncludeGoRoot:   settings.IncludeGoRoot,
		IgnoreFileRules: settings.IgnoreFileRules,
	}

	var err error
	dg.ListType, err = getDepGuardListType(settings.ListType)
	if err != nil {
		return nil, err
	}

	// if the list type was a blacklist the packages with error messages should be included in the blacklist package list
	if dg.ListType == depguard.LTBlacklist {
		noMessagePackages := make(map[string]bool)
		for _, pkg := range dg.Packages {
			noMessagePackages[pkg] = true
		}

		for pkg := range settings.PackagesWithErrorMessage {
			if _, ok := noMessagePackages[pkg]; !ok {
				dg.Packages = append(dg.Packages, pkg)
			}
		}
	}

	return &guardian{
		Depguard:             dg,
		pkgsWithErrorMessage: settings.PackagesWithErrorMessage,
	}, nil
}

func (g guardian) run(loadConfig *loader.Config, prog *loader.Program, pass *analysis.Pass) ([]goanalysis.Issue, error) {
	issues, err := g.Run(loadConfig, prog)
	if err != nil {
		return nil, err
	}

	res := make([]goanalysis.Issue, 0, len(issues))

	for _, issue := range issues {
		res = append(res,
			goanalysis.NewIssue(&result.Issue{
				Pos:        issue.Position,
				Text:       g.createMsg(issue.PackageName),
				FromLinter: depguardLinterName,
			}, pass),
		)
	}

	return res, nil
}

func (g guardian) createMsg(pkgName string) string {
	msgSuffix := "is in the blacklist"
	if g.ListType == depguard.LTWhitelist {
		msgSuffix = "is not in the whitelist"
	}

	var userSuppliedMsgSuffix string
	if g.pkgsWithErrorMessage != nil {
		userSuppliedMsgSuffix = g.pkgsWithErrorMessage[pkgName]
		if userSuppliedMsgSuffix != "" {
			userSuppliedMsgSuffix = ": " + userSuppliedMsgSuffix
		}
	}

	return fmt.Sprintf("%s %s%s", formatCode(pkgName, nil), msgSuffix, userSuppliedMsgSuffix)
}

func getDepGuardListType(listType string) (depguard.ListType, error) {
	if listType == "" {
		return depguard.LTBlacklist, nil
	}

	listT, found := depguard.StringToListType[strings.ToLower(listType)]
	if !found {
		return depguard.LTBlacklist, fmt.Errorf("unsure what list type %s is", listType)
	}

	return listT, nil
}
