package goanalysis

import (
	"fmt"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"

	"github.com/golangci/golangci-lint/internal/pkgcache"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/result"
	"github.com/golangci/golangci-lint/pkg/timeutils"
)

type runAnalyzersConfig interface {
	getName() string
	getLinterNameForDiagnostic(*Diagnostic) string
	getAnalyzers() []*analysis.Analyzer
	useOriginalPackages() bool
	reportIssues(*linter.Context) []Issue
	getLoadMode() LoadMode
}

func runAnalyzers(cfg runAnalyzersConfig, lintCtx *linter.Context) ([]result.Issue, error) {
	log := lintCtx.Log.Child("goanalysis")
	sw := timeutils.NewStopwatch("analyzers", log)

	const stagesToPrint = 10
	defer sw.PrintTopStages(stagesToPrint)

	runner := newRunner(cfg.getName(), log, lintCtx.PkgCache, lintCtx.LoadGuard, cfg.getLoadMode(), sw)

	pkgs := lintCtx.Packages
	if cfg.useOriginalPackages() {
		pkgs = lintCtx.OriginalPackages
	}

	issues, pkgsFromCache := loadIssuesFromCache(pkgs, lintCtx, cfg.getAnalyzers())
	var pkgsToAnalyze []*packages.Package
	for _, pkg := range pkgs {
		if !pkgsFromCache[pkg] {
			pkgsToAnalyze = append(pkgsToAnalyze, pkg)
		}
	}

	diags, errs, passToPkg := runner.run(cfg.getAnalyzers(), pkgsToAnalyze)

	defer func() {
		if len(errs) == 0 {
			// If we try to save to cache even if we have compilation errors
			// we won't see them on repeated runs.
			saveIssuesToCache(pkgs, pkgsFromCache, issues, lintCtx, cfg.getAnalyzers())
		}
	}()

	buildAllIssues := func() []result.Issue {
		var retIssues []result.Issue
		reportedIssues := cfg.reportIssues(lintCtx)
		for i := range reportedIssues {
			issue := &reportedIssues[i].Issue
			if issue.Pkg == nil {
				issue.Pkg = passToPkg[reportedIssues[i].Pass]
			}
			retIssues = append(retIssues, *issue)
		}
		retIssues = append(retIssues, buildIssues(diags, cfg.getLinterNameForDiagnostic)...)
		return retIssues
	}

	errIssues, err := buildIssuesFromIllTypedError(errs, lintCtx)
	if err != nil {
		return nil, err
	}

	issues = append(issues, errIssues...)
	issues = append(issues, buildAllIssues()...)

	return issues, nil
}

func buildIssues(diags []Diagnostic, linterNameBuilder func(diag *Diagnostic) string) []result.Issue {
	var issues []result.Issue
	for i := range diags {
		diag := &diags[i]
		linterName := linterNameBuilder(diag)

		var text string
		if diag.Analyzer.Name == linterName {
			text = diag.Message
		} else {
			text = fmt.Sprintf("%s: %s", diag.Analyzer.Name, diag.Message)
		}

		issues = append(issues, result.Issue{
			FromLinter: linterName,
			Text:       text,
			Pos:        diag.Position,
			Pkg:        diag.Pkg,
		})

		if len(diag.Related) > 0 {
			for _, info := range diag.Related {
				issues = append(issues, result.Issue{
					FromLinter: linterName,
					Text:       fmt.Sprintf("%s(related information): %s", diag.Analyzer.Name, info.Message),
					Pos:        diag.Pkg.Fset.Position(info.Pos),
					Pkg:        diag.Pkg,
				})
			}
		}
	}
	return issues
}

func getIssuesCacheKey(analyzers []*analysis.Analyzer) string {
	return "lint/result:" + analyzersHashID(analyzers)
}

func saveIssuesToCache(allPkgs []*packages.Package, pkgsFromCache map[*packages.Package]bool,
	issues []result.Issue, lintCtx *linter.Context, analyzers []*analysis.Analyzer) {
	startedAt := time.Now()
	perPkgIssues := map[*packages.Package][]result.Issue{}
	for ind := range issues {
		i := &issues[ind]
		perPkgIssues[i.Pkg] = append(perPkgIssues[i.Pkg], *i)
	}

	savedIssuesCount := int32(0)
	lintResKey := getIssuesCacheKey(analyzers)

	workerCount := runtime.GOMAXPROCS(-1)
	var wg sync.WaitGroup
	wg.Add(workerCount)

	pkgCh := make(chan *packages.Package, len(allPkgs))
	for i := 0; i < workerCount; i++ {
		go func() {
			defer wg.Done()
			for pkg := range pkgCh {
				pkgIssues := perPkgIssues[pkg]
				encodedIssues := make([]EncodingIssue, 0, len(pkgIssues))
				for ind := range pkgIssues {
					i := &pkgIssues[ind]
					encodedIssues = append(encodedIssues, EncodingIssue{
						FromLinter:           i.FromLinter,
						Text:                 i.Text,
						Pos:                  i.Pos,
						LineRange:            i.LineRange,
						Replacement:          i.Replacement,
						ExpectNoLint:         i.ExpectNoLint,
						ExpectedNoLintLinter: i.ExpectedNoLintLinter,
					})
				}

				atomic.AddInt32(&savedIssuesCount, int32(len(encodedIssues)))
				if err := lintCtx.PkgCache.Put(pkg, pkgcache.HashModeNeedAllDeps, lintResKey, encodedIssues); err != nil {
					lintCtx.Log.Infof("Failed to save package %s issues (%d) to cache: %s", pkg, len(pkgIssues), err)
				} else {
					issuesCacheDebugf("Saved package %s issues (%d) to cache", pkg, len(pkgIssues))
				}
			}
		}()
	}

	for _, pkg := range allPkgs {
		if pkgsFromCache[pkg] {
			continue
		}

		pkgCh <- pkg
	}
	close(pkgCh)
	wg.Wait()

	issuesCacheDebugf("Saved %d issues from %d packages to cache in %s", savedIssuesCount, len(allPkgs), time.Since(startedAt))
}

//nolint:gocritic
func loadIssuesFromCache(pkgs []*packages.Package, lintCtx *linter.Context,
	analyzers []*analysis.Analyzer) ([]result.Issue, map[*packages.Package]bool) {
	startedAt := time.Now()

	lintResKey := getIssuesCacheKey(analyzers)
	type cacheRes struct {
		issues  []result.Issue
		loadErr error
	}
	pkgToCacheRes := make(map[*packages.Package]*cacheRes, len(pkgs))
	for _, pkg := range pkgs {
		pkgToCacheRes[pkg] = &cacheRes{}
	}

	workerCount := runtime.GOMAXPROCS(-1)
	var wg sync.WaitGroup
	wg.Add(workerCount)

	pkgCh := make(chan *packages.Package, len(pkgs))
	for i := 0; i < workerCount; i++ {
		go func() {
			defer wg.Done()
			for pkg := range pkgCh {
				var pkgIssues []EncodingIssue
				err := lintCtx.PkgCache.Get(pkg, pkgcache.HashModeNeedAllDeps, lintResKey, &pkgIssues)
				cacheRes := pkgToCacheRes[pkg]
				cacheRes.loadErr = err
				if err != nil {
					continue
				}
				if len(pkgIssues) == 0 {
					continue
				}

				issues := make([]result.Issue, 0, len(pkgIssues))
				for _, i := range pkgIssues {
					issues = append(issues, result.Issue{
						FromLinter:           i.FromLinter,
						Text:                 i.Text,
						Pos:                  i.Pos,
						LineRange:            i.LineRange,
						Replacement:          i.Replacement,
						Pkg:                  pkg,
						ExpectNoLint:         i.ExpectNoLint,
						ExpectedNoLintLinter: i.ExpectedNoLintLinter,
					})
				}
				cacheRes.issues = issues
			}
		}()
	}

	for _, pkg := range pkgs {
		pkgCh <- pkg
	}
	close(pkgCh)
	wg.Wait()

	loadedIssuesCount := 0
	var issues []result.Issue
	pkgsFromCache := map[*packages.Package]bool{}
	for pkg, cacheRes := range pkgToCacheRes {
		if cacheRes.loadErr == nil {
			loadedIssuesCount += len(cacheRes.issues)
			pkgsFromCache[pkg] = true
			issues = append(issues, cacheRes.issues...)
			issuesCacheDebugf("Loaded package %s issues (%d) from cache", pkg, len(cacheRes.issues))
		} else {
			issuesCacheDebugf("Didn't load package %s issues from cache: %s", pkg, cacheRes.loadErr)
		}
	}
	issuesCacheDebugf("Loaded %d issues from cache in %s, analyzing %d/%d packages",
		loadedIssuesCount, time.Since(startedAt), len(pkgs)-len(pkgsFromCache), len(pkgs))
	return issues, pkgsFromCache
}

func analyzersHashID(analyzers []*analysis.Analyzer) string {
	names := make([]string, 0, len(analyzers))
	for _, a := range analyzers {
		names = append(names, a.Name)
	}

	sort.Strings(names)
	return strings.Join(names, ",")
}
