package goanalysis

import (
	"fmt"
	"go/types"
	"reflect"
	"runtime/debug"
	"time"

	"github.com/hashicorp/go-multierror"
	"github.com/pkg/errors"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/types/objectpath"

	"github.com/golangci/golangci-lint/internal/errorutil"
	"github.com/golangci/golangci-lint/internal/pkgcache"
)

type actionAllocator struct {
	allocatedActions []action
	nextFreeIndex    int
}

func newActionAllocator(maxCount int) *actionAllocator {
	return &actionAllocator{
		allocatedActions: make([]action, maxCount),
		nextFreeIndex:    0,
	}
}

func (actAlloc *actionAllocator) alloc() *action {
	if actAlloc.nextFreeIndex == len(actAlloc.allocatedActions) {
		panic(fmt.Sprintf("Made too many allocations of actions: %d allowed", len(actAlloc.allocatedActions)))
	}
	act := &actAlloc.allocatedActions[actAlloc.nextFreeIndex]
	actAlloc.nextFreeIndex++
	return act
}

// An action represents one unit of analysis work: the application of
// one analysis to one package. Actions form a DAG, both within a
// package (as different analyzers are applied, either in sequence or
// parallel), and across packages (as dependencies are analyzed).
type action struct {
	a                   *analysis.Analyzer
	pkg                 *packages.Package
	pass                *analysis.Pass
	deps                []*action
	objectFacts         map[objectFactKey]analysis.Fact
	packageFacts        map[packageFactKey]analysis.Fact
	result              interface{}
	diagnostics         []analysis.Diagnostic
	err                 error
	r                   *runner
	analysisDoneCh      chan struct{}
	loadCachedFactsDone bool
	loadCachedFactsOk   bool
	isroot              bool
	isInitialPkg        bool
	needAnalyzeSource   bool
}

func (act *action) String() string {
	return fmt.Sprintf("%s@%s", act.a, act.pkg)
}

func (act *action) loadCachedFacts() bool {
	if act.loadCachedFactsDone { // can't be set in parallel
		return act.loadCachedFactsOk
	}

	res := func() bool {
		if act.isInitialPkg {
			return true // load cached facts only for non-initial packages
		}

		if len(act.a.FactTypes) == 0 {
			return true // no need to load facts
		}

		return act.loadPersistedFacts()
	}()
	act.loadCachedFactsDone = true
	act.loadCachedFactsOk = res
	return res
}

func (act *action) waitUntilDependingAnalyzersWorked() {
	for _, dep := range act.deps {
		if dep.pkg == act.pkg {
			<-dep.analysisDoneCh
		}
	}
}

func (act *action) analyzeSafe() {
	defer func() {
		if p := recover(); p != nil {
			act.err = errorutil.NewPanicError(fmt.Sprintf("%s: package %q (isInitialPkg: %t, needAnalyzeSource: %t): %s",
				act.a.Name, act.pkg.Name, act.isInitialPkg, act.needAnalyzeSource, p), debug.Stack())
		}
	}()
	act.r.sw.TrackStage(act.a.Name, func() {
		act.analyze()
	})
}

func (act *action) analyze() {
	defer close(act.analysisDoneCh) // unblock actions depending on this action

	if !act.needAnalyzeSource {
		return
	}

	defer func(now time.Time) {
		analyzeDebugf("go/analysis: %s: %s: analyzed package %q in %s", act.r.prefix, act.a.Name, act.pkg.Name, time.Since(now))
	}(time.Now())

	// Report an error if any dependency failures.
	var depErrors *multierror.Error
	for _, dep := range act.deps {
		if dep.err == nil {
			continue
		}

		depErrors = multierror.Append(depErrors, errors.Cause(dep.err))
	}
	if depErrors != nil {
		depErrors.ErrorFormat = func(e []error) string {
			return fmt.Sprintf("failed prerequisites: %v", e)
		}

		act.err = depErrors
		return
	}

	// Plumb the output values of the dependencies
	// into the inputs of this action.  Also facts.
	inputs := make(map[*analysis.Analyzer]interface{})
	startedAt := time.Now()
	for _, dep := range act.deps {
		if dep.pkg == act.pkg {
			// Same package, different analysis (horizontal edge):
			// in-memory outputs of prerequisite analyzers
			// become inputs to this analysis pass.
			inputs[dep.a] = dep.result
		} else if dep.a == act.a { // (always true)
			// Same analysis, different package (vertical edge):
			// serialized facts produced by prerequisite analysis
			// become available to this analysis pass.
			inheritFacts(act, dep)
		}
	}
	factsDebugf("%s: Inherited facts in %s", act, time.Since(startedAt))

	// Run the analysis.
	pass := &analysis.Pass{
		Analyzer:          act.a,
		Fset:              act.pkg.Fset,
		Files:             act.pkg.Syntax,
		OtherFiles:        act.pkg.OtherFiles,
		Pkg:               act.pkg.Types,
		TypesInfo:         act.pkg.TypesInfo,
		TypesSizes:        act.pkg.TypesSizes,
		ResultOf:          inputs,
		Report:            func(d analysis.Diagnostic) { act.diagnostics = append(act.diagnostics, d) },
		ImportObjectFact:  act.importObjectFact,
		ExportObjectFact:  act.exportObjectFact,
		ImportPackageFact: act.importPackageFact,
		ExportPackageFact: act.exportPackageFact,
		AllObjectFacts:    act.allObjectFacts,
		AllPackageFacts:   act.allPackageFacts,
	}
	act.pass = pass
	act.r.passToPkgGuard.Lock()
	act.r.passToPkg[pass] = act.pkg
	act.r.passToPkgGuard.Unlock()

	if act.pkg.IllTyped {
		// It looks like there should be !pass.Analyzer.RunDespiteErrors
		// but govet's cgocall crashes on it. Govet itself contains !pass.Analyzer.RunDespiteErrors condition here,
		// but it exits before it if packages.Load have failed.
		act.err = errors.Wrap(&IllTypedError{Pkg: act.pkg}, "analysis skipped")
	} else {
		startedAt = time.Now()
		act.result, act.err = pass.Analyzer.Run(pass)
		analyzedIn := time.Since(startedAt)
		if analyzedIn > time.Millisecond*10 {
			debugf("%s: run analyzer in %s", act, analyzedIn)
		}
	}

	// disallow calls after Run
	pass.ExportObjectFact = nil
	pass.ExportPackageFact = nil

	if err := act.persistFactsToCache(); err != nil {
		act.r.log.Warnf("Failed to persist facts to cache: %s", err)
	}
}

// importObjectFact implements Pass.ImportObjectFact.
// Given a non-nil pointer ptr of type *T, where *T satisfies Fact,
// importObjectFact copies the fact value to *ptr.
func (act *action) importObjectFact(obj types.Object, ptr analysis.Fact) bool {
	if obj == nil {
		panic("nil object")
	}
	key := objectFactKey{obj, act.factType(ptr)}
	if v, ok := act.objectFacts[key]; ok {
		reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
		return true
	}
	return false
}

// exportObjectFact implements Pass.ExportObjectFact.
func (act *action) exportObjectFact(obj types.Object, fact analysis.Fact) {
	if obj.Pkg() != act.pkg.Types {
		act.r.log.Panicf("internal error: in analysis %s of package %s: Fact.Set(%s, %T): can't set facts on objects belonging another package",
			act.a, act.pkg, obj, fact)
	}

	key := objectFactKey{obj, act.factType(fact)}
	act.objectFacts[key] = fact // clobber any existing entry
	if isFactsExportDebug {
		objstr := types.ObjectString(obj, (*types.Package).Name)
		factsExportDebugf("%s: object %s has fact %s\n",
			act.pkg.Fset.Position(obj.Pos()), objstr, fact)
	}
}

func (act *action) allObjectFacts() []analysis.ObjectFact {
	out := make([]analysis.ObjectFact, 0, len(act.objectFacts))
	for key, fact := range act.objectFacts {
		out = append(out, analysis.ObjectFact{
			Object: key.obj,
			Fact:   fact,
		})
	}
	return out
}

// importPackageFact implements Pass.ImportPackageFact.
// Given a non-nil pointer ptr of type *T, where *T satisfies Fact,
// fact copies the fact value to *ptr.
func (act *action) importPackageFact(pkg *types.Package, ptr analysis.Fact) bool {
	if pkg == nil {
		panic("nil package")
	}
	key := packageFactKey{pkg, act.factType(ptr)}
	if v, ok := act.packageFacts[key]; ok {
		reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
		return true
	}
	return false
}

// exportPackageFact implements Pass.ExportPackageFact.
func (act *action) exportPackageFact(fact analysis.Fact) {
	key := packageFactKey{act.pass.Pkg, act.factType(fact)}
	act.packageFacts[key] = fact // clobber any existing entry
	factsDebugf("%s: package %s has fact %s\n",
		act.pkg.Fset.Position(act.pass.Files[0].Pos()), act.pass.Pkg.Path(), fact)
}

func (act *action) allPackageFacts() []analysis.PackageFact {
	out := make([]analysis.PackageFact, 0, len(act.packageFacts))
	for key, fact := range act.packageFacts {
		out = append(out, analysis.PackageFact{
			Package: key.pkg,
			Fact:    fact,
		})
	}
	return out
}

func (act *action) factType(fact analysis.Fact) reflect.Type {
	t := reflect.TypeOf(fact)
	if t.Kind() != reflect.Ptr {
		act.r.log.Fatalf("invalid Fact type: got %T, want pointer", t)
	}
	return t
}

func (act *action) persistFactsToCache() error {
	analyzer := act.a
	if len(analyzer.FactTypes) == 0 {
		return nil
	}

	// Merge new facts into the package and persist them.
	var facts []Fact
	for key, fact := range act.packageFacts {
		if key.pkg != act.pkg.Types {
			// The fact is from inherited facts from another package
			continue
		}
		facts = append(facts, Fact{
			Path: "",
			Fact: fact,
		})
	}
	for key, fact := range act.objectFacts {
		obj := key.obj
		if obj.Pkg() != act.pkg.Types {
			// The fact is from inherited facts from another package
			continue
		}

		path, err := objectpath.For(obj)
		if err != nil {
			// The object is not globally addressable
			continue
		}

		facts = append(facts, Fact{
			Path: string(path),
			Fact: fact,
		})
	}

	factsCacheDebugf("Caching %d facts for package %q and analyzer %s", len(facts), act.pkg.Name, act.a.Name)

	key := fmt.Sprintf("%s/facts", analyzer.Name)
	return act.r.pkgCache.Put(act.pkg, pkgcache.HashModeNeedAllDeps, key, facts)
}

func (act *action) loadPersistedFacts() bool {
	var facts []Fact
	key := fmt.Sprintf("%s/facts", act.a.Name)
	if err := act.r.pkgCache.Get(act.pkg, pkgcache.HashModeNeedAllDeps, key, &facts); err != nil {
		if err != pkgcache.ErrMissing {
			act.r.log.Warnf("Failed to get persisted facts: %s", err)
		}

		factsCacheDebugf("No cached facts for package %q and analyzer %s", act.pkg.Name, act.a.Name)
		return false
	}

	factsCacheDebugf("Loaded %d cached facts for package %q and analyzer %s", len(facts), act.pkg.Name, act.a.Name)

	for _, f := range facts {
		if f.Path == "" { // this is a package fact
			key := packageFactKey{act.pkg.Types, act.factType(f.Fact)}
			act.packageFacts[key] = f.Fact
			continue
		}
		obj, err := objectpath.Object(act.pkg.Types, objectpath.Path(f.Path))
		if err != nil {
			// Be lenient about these errors. For example, when
			// analyzing io/ioutil from source, we may get a fact
			// for methods on the devNull type, and objectpath
			// will happily create a path for them. However, when
			// we later load io/ioutil from export data, the path
			// no longer resolves.
			//
			// If an exported type embeds the unexported type,
			// then (part of) the unexported type will become part
			// of the type information and our path will resolve
			// again.
			continue
		}
		factKey := objectFactKey{obj, act.factType(f.Fact)}
		act.objectFacts[factKey] = f.Fact
	}

	return true
}

func (act *action) markDepsForAnalyzingSource() {
	// Horizontal deps (analyzer.Requires) must be loaded from source and analyzed before analyzing
	// this action.
	for _, dep := range act.deps {
		if dep.pkg == act.pkg {
			// Analyze source only for horizontal dependencies, e.g. from "buildssa".
			dep.needAnalyzeSource = true // can't be set in parallel
		}
	}
}
