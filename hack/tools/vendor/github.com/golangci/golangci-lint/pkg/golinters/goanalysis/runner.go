// checker is a partial copy of https://github.com/golang/tools/blob/master/go/analysis/internal/checker
// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package goanalysis defines the implementation of the checker commands.
// The same code drives the multi-analysis driver, the single-analysis
// driver that is conventionally provided for convenience along with
// each analysis package, and the test driver.
package goanalysis

import (
	"encoding/gob"
	"go/token"
	"runtime"
	"sort"
	"sync"

	"github.com/pkg/errors"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"

	"github.com/golangci/golangci-lint/internal/errorutil"
	"github.com/golangci/golangci-lint/internal/pkgcache"
	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis/load"
	"github.com/golangci/golangci-lint/pkg/logutils"
	"github.com/golangci/golangci-lint/pkg/timeutils"
)

var (
	debugf = logutils.Debug("goanalysis")

	analyzeDebugf     = logutils.Debug("goanalysis/analyze")
	isMemoryDebug     = logutils.HaveDebugTag("goanalysis/memory")
	issuesCacheDebugf = logutils.Debug("goanalysis/issues/cache")

	factsDebugf        = logutils.Debug("goanalysis/facts")
	factsCacheDebugf   = logutils.Debug("goanalysis/facts/cache")
	factsInheritDebugf = logutils.Debug("goanalysis/facts/inherit")
	factsExportDebugf  = logutils.Debug("goanalysis/facts")
	isFactsExportDebug = logutils.HaveDebugTag("goanalysis/facts/export")
)

type Diagnostic struct {
	analysis.Diagnostic
	Analyzer *analysis.Analyzer
	Position token.Position
	Pkg      *packages.Package
}

type runner struct {
	log            logutils.Log
	prefix         string // ensure unique analyzer names
	pkgCache       *pkgcache.Cache
	loadGuard      *load.Guard
	loadMode       LoadMode
	passToPkg      map[*analysis.Pass]*packages.Package
	passToPkgGuard sync.Mutex
	sw             *timeutils.Stopwatch
}

func newRunner(prefix string, logger logutils.Log, pkgCache *pkgcache.Cache, loadGuard *load.Guard,
	loadMode LoadMode, sw *timeutils.Stopwatch) *runner {
	return &runner{
		prefix:    prefix,
		log:       logger,
		pkgCache:  pkgCache,
		loadGuard: loadGuard,
		loadMode:  loadMode,
		passToPkg: map[*analysis.Pass]*packages.Package{},
		sw:        sw,
	}
}

// Run loads the packages specified by args using go/packages,
// then applies the specified analyzers to them.
// Analysis flags must already have been set.
// It provides most of the logic for the main functions of both the
// singlechecker and the multi-analysis commands.
// It returns the appropriate exit code.
func (r *runner) run(analyzers []*analysis.Analyzer, initialPackages []*packages.Package) ([]Diagnostic,
	[]error, map[*analysis.Pass]*packages.Package) {
	debugf("Analyzing %d packages on load mode %s", len(initialPackages), r.loadMode)
	defer r.pkgCache.Trim()

	roots := r.analyze(initialPackages, analyzers)

	diags, errs := extractDiagnostics(roots)

	return diags, errs, r.passToPkg
}

type actKey struct {
	*analysis.Analyzer
	*packages.Package
}

func (r *runner) markAllActions(a *analysis.Analyzer, pkg *packages.Package, markedActions map[actKey]struct{}) {
	k := actKey{a, pkg}
	if _, ok := markedActions[k]; ok {
		return
	}

	for _, req := range a.Requires {
		r.markAllActions(req, pkg, markedActions)
	}

	if len(a.FactTypes) != 0 {
		for path := range pkg.Imports {
			r.markAllActions(a, pkg.Imports[path], markedActions)
		}
	}

	markedActions[k] = struct{}{}
}

func (r *runner) makeAction(a *analysis.Analyzer, pkg *packages.Package,
	initialPkgs map[*packages.Package]bool, actions map[actKey]*action, actAlloc *actionAllocator) *action {
	k := actKey{a, pkg}
	act, ok := actions[k]
	if ok {
		return act
	}

	act = actAlloc.alloc()
	act.a = a
	act.pkg = pkg
	act.r = r
	act.isInitialPkg = initialPkgs[pkg]
	act.needAnalyzeSource = initialPkgs[pkg]
	act.analysisDoneCh = make(chan struct{})

	depsCount := len(a.Requires)
	if len(a.FactTypes) > 0 {
		depsCount += len(pkg.Imports)
	}
	act.deps = make([]*action, 0, depsCount)

	// Add a dependency on each required analyzers.
	for _, req := range a.Requires {
		act.deps = append(act.deps, r.makeAction(req, pkg, initialPkgs, actions, actAlloc))
	}

	r.buildActionFactDeps(act, a, pkg, initialPkgs, actions, actAlloc)

	actions[k] = act

	return act
}

func (r *runner) buildActionFactDeps(act *action, a *analysis.Analyzer, pkg *packages.Package,
	initialPkgs map[*packages.Package]bool, actions map[actKey]*action, actAlloc *actionAllocator) {
	// An analysis that consumes/produces facts
	// must run on the package's dependencies too.
	if len(a.FactTypes) == 0 {
		return
	}

	act.objectFacts = make(map[objectFactKey]analysis.Fact)
	act.packageFacts = make(map[packageFactKey]analysis.Fact)

	paths := make([]string, 0, len(pkg.Imports))
	for path := range pkg.Imports {
		paths = append(paths, path)
	}
	sort.Strings(paths) // for determinism
	for _, path := range paths {
		dep := r.makeAction(a, pkg.Imports[path], initialPkgs, actions, actAlloc)
		act.deps = append(act.deps, dep)
	}

	// Need to register fact types for pkgcache proper gob encoding.
	for _, f := range a.FactTypes {
		gob.Register(f)
	}
}

//nolint:gocritic
func (r *runner) prepareAnalysis(pkgs []*packages.Package,
	analyzers []*analysis.Analyzer) (map[*packages.Package]bool, []*action, []*action) {
	// Construct the action graph.

	// Each graph node (action) is one unit of analysis.
	// Edges express package-to-package (vertical) dependencies,
	// and analysis-to-analysis (horizontal) dependencies.

	// This place is memory-intensive: e.g. Istio project has 120k total actions.
	// Therefore, optimize it carefully.
	markedActions := make(map[actKey]struct{}, len(analyzers)*len(pkgs))
	for _, a := range analyzers {
		for _, pkg := range pkgs {
			r.markAllActions(a, pkg, markedActions)
		}
	}
	totalActionsCount := len(markedActions)

	actions := make(map[actKey]*action, totalActionsCount)
	actAlloc := newActionAllocator(totalActionsCount)

	initialPkgs := make(map[*packages.Package]bool, len(pkgs))
	for _, pkg := range pkgs {
		initialPkgs[pkg] = true
	}

	// Build nodes for initial packages.
	roots := make([]*action, 0, len(pkgs)*len(analyzers))
	for _, a := range analyzers {
		for _, pkg := range pkgs {
			root := r.makeAction(a, pkg, initialPkgs, actions, actAlloc)
			root.isroot = true
			roots = append(roots, root)
		}
	}

	allActions := make([]*action, 0, len(actions))
	for _, act := range actions {
		allActions = append(allActions, act)
	}

	debugf("Built %d actions", len(actions))

	return initialPkgs, allActions, roots
}

func (r *runner) analyze(pkgs []*packages.Package, analyzers []*analysis.Analyzer) []*action {
	initialPkgs, actions, rootActions := r.prepareAnalysis(pkgs, analyzers)

	actionPerPkg := map[*packages.Package][]*action{}
	for _, act := range actions {
		actionPerPkg[act.pkg] = append(actionPerPkg[act.pkg], act)
	}

	// Fill Imports field.
	loadingPackages := map[*packages.Package]*loadingPackage{}
	var dfs func(pkg *packages.Package)
	dfs = func(pkg *packages.Package) {
		if loadingPackages[pkg] != nil {
			return
		}

		imports := map[string]*loadingPackage{}
		for impPath, imp := range pkg.Imports {
			dfs(imp)
			impLp := loadingPackages[imp]
			impLp.dependents++
			imports[impPath] = impLp
		}

		loadingPackages[pkg] = &loadingPackage{
			pkg:        pkg,
			imports:    imports,
			isInitial:  initialPkgs[pkg],
			log:        r.log,
			actions:    actionPerPkg[pkg],
			loadGuard:  r.loadGuard,
			dependents: 1, // self dependent
		}
	}
	for _, act := range actions {
		dfs(act.pkg)
	}

	// Limit memory and IO usage.
	gomaxprocs := runtime.GOMAXPROCS(-1)
	debugf("Analyzing at most %d packages in parallel", gomaxprocs)
	loadSem := make(chan struct{}, gomaxprocs)

	var wg sync.WaitGroup
	debugf("There are %d initial and %d total packages", len(initialPkgs), len(loadingPackages))
	for _, lp := range loadingPackages {
		if lp.isInitial {
			wg.Add(1)
			go func(lp *loadingPackage) {
				lp.analyzeRecursive(r.loadMode, loadSem)
				wg.Done()
			}(lp)
		}
	}
	wg.Wait()

	return rootActions
}

//nolint:nakedret
func extractDiagnostics(roots []*action) (retDiags []Diagnostic, retErrors []error) {
	extracted := make(map[*action]bool)
	var extract func(*action)
	var visitAll func(actions []*action)
	visitAll = func(actions []*action) {
		for _, act := range actions {
			if !extracted[act] {
				extracted[act] = true
				visitAll(act.deps)
				extract(act)
			}
		}
	}

	// De-duplicate diagnostics by position (not token.Pos) to
	// avoid double-reporting in source files that belong to
	// multiple packages, such as foo and foo.test.
	type key struct {
		token.Position
		*analysis.Analyzer
		message string
	}
	seen := make(map[key]bool)

	extract = func(act *action) {
		if act.err != nil {
			if pe, ok := act.err.(*errorutil.PanicError); ok {
				panic(pe)
			}
			retErrors = append(retErrors, errors.Wrap(act.err, act.a.Name))
			return
		}

		if act.isroot {
			for _, diag := range act.diagnostics {
				// We don't display a.Name/f.Category
				// as most users don't care.

				posn := act.pkg.Fset.Position(diag.Pos)
				k := key{posn, act.a, diag.Message}
				if seen[k] {
					continue // duplicate
				}
				seen[k] = true

				retDiag := Diagnostic{
					Diagnostic: diag,
					Analyzer:   act.a,
					Position:   posn,
					Pkg:        act.pkg,
				}
				retDiags = append(retDiags, retDiag)
			}
		}
	}
	visitAll(roots)
	return
}
