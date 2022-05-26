package ruleguard

import (
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"go/types"
	"io"
	"io/ioutil"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/quasilyte/go-ruleguard/internal/goenv"
	"github.com/quasilyte/go-ruleguard/internal/stdinfo"
	"github.com/quasilyte/go-ruleguard/ruleguard/ir"
	"github.com/quasilyte/go-ruleguard/ruleguard/quasigo"
	"github.com/quasilyte/go-ruleguard/ruleguard/typematch"
)

type engine struct {
	state *engineState

	ruleSet *goRuleSet
}

func newEngine() *engine {
	return &engine{
		state: newEngineState(),
	}
}

func (e *engine) LoadedGroups() []GoRuleGroup {
	result := make([]GoRuleGroup, 0, len(e.ruleSet.groups))
	for _, g := range e.ruleSet.groups {
		result = append(result, *g)
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Name < result[j].Name
	})
	return result
}

func (e *engine) Load(ctx *LoadContext, buildContext *build.Context, filename string, r io.Reader) error {
	data, err := ioutil.ReadAll(r)
	if err != nil {
		return err
	}
	imp := newGoImporter(e.state, goImporterConfig{
		fset:         ctx.Fset,
		debugImports: ctx.DebugImports,
		debugPrint:   ctx.DebugPrint,
		buildContext: buildContext,
	})
	irfile, pkg, err := convertAST(ctx, imp, filename, data)
	if err != nil {
		return err
	}
	config := irLoaderConfig{
		state:      e.state,
		pkg:        pkg,
		ctx:        ctx,
		importer:   imp,
		itab:       typematch.NewImportsTab(stdinfo.Packages),
		gogrepFset: token.NewFileSet(),
	}
	l := newIRLoader(config)
	rset, err := l.LoadFile(filename, irfile)
	if err != nil {
		return err
	}

	if e.ruleSet == nil {
		e.ruleSet = rset
	} else {
		combinedRuleSet, err := mergeRuleSets([]*goRuleSet{e.ruleSet, rset})
		if err != nil {
			return err
		}
		e.ruleSet = combinedRuleSet
	}

	return nil
}

func (e *engine) LoadFromIR(ctx *LoadContext, buildContext *build.Context, filename string, f *ir.File) error {
	imp := newGoImporter(e.state, goImporterConfig{
		fset:         ctx.Fset,
		debugImports: ctx.DebugImports,
		debugPrint:   ctx.DebugPrint,
		buildContext: buildContext,
	})
	config := irLoaderConfig{
		state:      e.state,
		ctx:        ctx,
		importer:   imp,
		itab:       typematch.NewImportsTab(stdinfo.Packages),
		gogrepFset: token.NewFileSet(),
	}
	l := newIRLoader(config)
	rset, err := l.LoadFile(filename, f)
	if err != nil {
		return err
	}

	if e.ruleSet == nil {
		e.ruleSet = rset
	} else {
		combinedRuleSet, err := mergeRuleSets([]*goRuleSet{e.ruleSet, rset})
		if err != nil {
			return err
		}
		e.ruleSet = combinedRuleSet
	}

	return nil
}

func (e *engine) Run(ctx *RunContext, buildContext *build.Context, f *ast.File) error {
	if e.ruleSet == nil {
		return errors.New("used Run() with an empty rule set; forgot to call Load() first?")
	}
	rset := e.ruleSet
	return newRulesRunner(ctx, buildContext, e.state, rset).run(f)
}

// engineState is a shared state inside the engine.
type engineState struct {
	env *quasigo.Env

	typeByFQNMu sync.RWMutex
	typeByFQN   map[string]types.Type

	pkgCacheMu sync.RWMutex
	// pkgCache contains all imported packages, from any importer.
	pkgCache map[string]*types.Package
}

func newEngineState() *engineState {
	env := quasigo.NewEnv()
	state := &engineState{
		env:       env,
		pkgCache:  make(map[string]*types.Package),
		typeByFQN: map[string]types.Type{},
	}
	for key, typ := range typeByName {
		state.typeByFQN[key] = typ
	}
	initEnv(state, env)
	return state
}

func (state *engineState) GetCachedPackage(pkgPath string) *types.Package {
	state.pkgCacheMu.RLock()
	pkg := state.pkgCache[pkgPath]
	state.pkgCacheMu.RUnlock()
	return pkg
}

func (state *engineState) AddCachedPackage(pkgPath string, pkg *types.Package) {
	state.pkgCacheMu.Lock()
	state.addCachedPackage(pkgPath, pkg)
	state.pkgCacheMu.Unlock()
}

func (state *engineState) addCachedPackage(pkgPath string, pkg *types.Package) {
	state.pkgCache[pkgPath] = pkg

	// Also add all complete packages that are dependencies of the pkg.
	// This way we cache more and avoid duplicated package loading
	// which can lead to typechecking issues.
	//
	// Note that it does not increase our memory consumption
	// as these packages are reachable via pkg, so they'll
	// not be freed by GC anyway.
	for _, imported := range pkg.Imports() {
		if imported.Complete() {
			state.addCachedPackage(imported.Path(), imported)
		}
	}
}

func (state *engineState) FindType(importer *goImporter, currentPkg *types.Package, fqn string) (types.Type, error) {
	// TODO(quasilyte): we can pre-populate the cache during the Load() phase.
	// If we inspect the AST of a user function, all constant FQN can be preloaded.
	// It could be a good thing as Load() is not expected to be executed in
	// concurrent environment, so write-locking is not a big deal there.

	state.typeByFQNMu.RLock()
	cachedType, ok := state.typeByFQN[fqn]
	state.typeByFQNMu.RUnlock()
	if ok {
		return cachedType, nil
	}

	// Code below is under a write critical section.
	state.typeByFQNMu.Lock()
	defer state.typeByFQNMu.Unlock()

	typ, err := state.findTypeNoCache(importer, currentPkg, fqn)
	if err != nil {
		return nil, err
	}
	state.typeByFQN[fqn] = typ
	return typ, nil
}

func (state *engineState) findTypeNoCache(importer *goImporter, currentPkg *types.Package, fqn string) (types.Type, error) {
	pos := strings.LastIndexByte(fqn, '.')
	if pos == -1 {
		return nil, fmt.Errorf("%s is not a valid FQN", fqn)
	}
	pkgPath := fqn[:pos]
	objectName := fqn[pos+1:]
	var pkg *types.Package
	if currentPkg != nil {
		if directDep := findDependency(currentPkg, pkgPath); directDep != nil {
			pkg = directDep
		}
	}
	if pkg == nil {
		loadedPkg, err := importer.Import(pkgPath)
		if err != nil {
			return nil, err
		}
		pkg = loadedPkg
	}
	obj := pkg.Scope().Lookup(objectName)
	if obj == nil {
		return nil, fmt.Errorf("%s is not found in %s", objectName, pkgPath)
	}
	typ := obj.Type()
	state.typeByFQN[fqn] = typ
	return typ, nil
}

func inferBuildContext() *build.Context {
	// Inherit most fields from the build.Default.
	ctx := build.Default

	env, err := goenv.Read()
	if err != nil {
		return &ctx
	}

	ctx.GOROOT = env["GOROOT"]
	ctx.GOPATH = env["GOPATH"]
	ctx.GOARCH = env["GOARCH"]
	ctx.GOOS = env["GOOS"]

	switch os.Getenv("CGO_ENABLED") {
	case "0":
		ctx.CgoEnabled = false
	case "1":
		ctx.CgoEnabled = true
	default:
		ctx.CgoEnabled = env["CGO_ENABLED"] == "1"
	}

	return &ctx
}
