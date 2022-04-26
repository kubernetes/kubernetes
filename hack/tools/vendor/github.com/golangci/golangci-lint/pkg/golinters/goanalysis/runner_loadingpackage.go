package goanalysis

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/types"
	"os"
	"reflect"
	"sync"
	"sync/atomic"

	"github.com/pkg/errors"
	"golang.org/x/tools/go/gcexportdata"
	"golang.org/x/tools/go/packages"

	"github.com/golangci/golangci-lint/pkg/golinters/goanalysis/load"
	"github.com/golangci/golangci-lint/pkg/logutils"
)

const unsafePkgName = "unsafe"

type loadingPackage struct {
	pkg         *packages.Package
	imports     map[string]*loadingPackage
	isInitial   bool
	log         logutils.Log
	actions     []*action // all actions with this package
	loadGuard   *load.Guard
	dependents  int32 // number of depending on it packages
	analyzeOnce sync.Once
	decUseMutex sync.Mutex
}

func (lp *loadingPackage) analyzeRecursive(loadMode LoadMode, loadSem chan struct{}) {
	lp.analyzeOnce.Do(func() {
		// Load the direct dependencies, in parallel.
		var wg sync.WaitGroup
		wg.Add(len(lp.imports))
		for _, imp := range lp.imports {
			go func(imp *loadingPackage) {
				imp.analyzeRecursive(loadMode, loadSem)
				wg.Done()
			}(imp)
		}
		wg.Wait()
		lp.analyze(loadMode, loadSem)
	})
}

func (lp *loadingPackage) analyze(loadMode LoadMode, loadSem chan struct{}) {
	loadSem <- struct{}{}
	defer func() {
		<-loadSem
	}()

	// Save memory on unused more fields.
	defer lp.decUse(loadMode < LoadModeWholeProgram)

	if err := lp.loadWithFacts(loadMode); err != nil {
		werr := errors.Wrapf(err, "failed to load package %s", lp.pkg.Name)
		// Don't need to write error to errCh, it will be extracted and reported on another layer.
		// Unblock depending on actions and propagate error.
		for _, act := range lp.actions {
			close(act.analysisDoneCh)
			act.err = werr
		}
		return
	}

	var actsWg sync.WaitGroup
	actsWg.Add(len(lp.actions))
	for _, act := range lp.actions {
		go func(act *action) {
			defer actsWg.Done()

			act.waitUntilDependingAnalyzersWorked()

			act.analyzeSafe()
		}(act)
	}
	actsWg.Wait()
}

func (lp *loadingPackage) loadFromSource(loadMode LoadMode) error {
	pkg := lp.pkg

	// Many packages have few files, much fewer than there
	// are CPU cores. Additionally, parsing each individual file is
	// very fast. A naive parallel implementation of this loop won't
	// be faster, and tends to be slower due to extra scheduling,
	// bookkeeping and potentially false sharing of cache lines.
	pkg.Syntax = make([]*ast.File, 0, len(pkg.CompiledGoFiles))
	for _, file := range pkg.CompiledGoFiles {
		f, err := parser.ParseFile(pkg.Fset, file, nil, parser.ParseComments)
		if err != nil {
			pkg.Errors = append(pkg.Errors, lp.convertError(err)...)
			continue
		}
		pkg.Syntax = append(pkg.Syntax, f)
	}
	if len(pkg.Errors) != 0 {
		pkg.IllTyped = true
		return nil
	}

	if loadMode == LoadModeSyntax {
		return nil
	}

	// Call NewPackage directly with explicit name.
	// This avoids skew between golist and go/types when the files'
	// package declarations are inconsistent.
	// Subtle: we populate all Types fields with an empty Package
	// before loading export data so that export data processing
	// never has to create a types.Package for an indirect dependency,
	// which would then require that such created packages be explicitly
	// inserted back into the Import graph as a final step after export data loading.
	pkg.Types = types.NewPackage(pkg.PkgPath, pkg.Name)

	pkg.IllTyped = true

	pkg.TypesInfo = &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Scopes:     make(map[ast.Node]*types.Scope),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}

	importer := func(path string) (*types.Package, error) {
		if path == unsafePkgName {
			return types.Unsafe, nil
		}
		if path == "C" {
			// go/packages doesn't tell us that cgo preprocessing
			// failed. When we subsequently try to parse the package,
			// we'll encounter the raw C import.
			return nil, errors.New("cgo preprocessing failed")
		}
		imp := pkg.Imports[path]
		if imp == nil {
			return nil, nil
		}
		if len(imp.Errors) > 0 {
			return nil, imp.Errors[0]
		}
		return imp.Types, nil
	}
	tc := &types.Config{
		Importer: importerFunc(importer),
		Error: func(err error) {
			pkg.Errors = append(pkg.Errors, lp.convertError(err)...)
		},
	}
	_ = types.NewChecker(tc, pkg.Fset, pkg.Types, pkg.TypesInfo).Files(pkg.Syntax)
	// Don't handle error here: errors are adding by tc.Error function.

	illTyped := len(pkg.Errors) != 0
	if !illTyped {
		for _, imp := range lp.imports {
			if imp.pkg.IllTyped {
				illTyped = true
				break
			}
		}
	}
	pkg.IllTyped = illTyped
	return nil
}

func (lp *loadingPackage) loadFromExportData() error {
	pkg := lp.pkg

	// Call NewPackage directly with explicit name.
	// This avoids skew between golist and go/types when the files'
	// package declarations are inconsistent.
	// Subtle: we populate all Types fields with an empty Package
	// before loading export data so that export data processing
	// never has to create a types.Package for an indirect dependency,
	// which would then require that such created packages be explicitly
	// inserted back into the Import graph as a final step after export data loading.
	pkg.Types = types.NewPackage(pkg.PkgPath, pkg.Name)

	pkg.IllTyped = true
	for path, pkg := range pkg.Imports {
		if pkg.Types == nil {
			return fmt.Errorf("dependency %q hasn't been loaded yet", path)
		}
	}
	if pkg.ExportFile == "" {
		return fmt.Errorf("no export data for %q", pkg.ID)
	}
	f, err := os.Open(pkg.ExportFile)
	if err != nil {
		return err
	}
	defer f.Close()

	r, err := gcexportdata.NewReader(f)
	if err != nil {
		return err
	}

	view := make(map[string]*types.Package)  // view seen by gcexportdata
	seen := make(map[*packages.Package]bool) // all visited packages
	var visit func(pkgs map[string]*packages.Package)
	visit = func(pkgs map[string]*packages.Package) {
		for _, pkg := range pkgs {
			if !seen[pkg] {
				seen[pkg] = true
				view[pkg.PkgPath] = pkg.Types
				visit(pkg.Imports)
			}
		}
	}
	visit(pkg.Imports)
	tpkg, err := gcexportdata.Read(r, pkg.Fset, view, pkg.PkgPath)
	if err != nil {
		return err
	}
	pkg.Types = tpkg
	pkg.IllTyped = false
	return nil
}

func (lp *loadingPackage) loadWithFacts(loadMode LoadMode) error {
	pkg := lp.pkg

	if pkg.PkgPath == unsafePkgName {
		// Fill in the blanks to avoid surprises.
		pkg.Syntax = []*ast.File{}
		if loadMode >= LoadModeTypesInfo {
			pkg.Types = types.Unsafe
			pkg.TypesInfo = new(types.Info)
		}
		return nil
	}

	if pkg.TypesInfo != nil {
		// Already loaded package, e.g. because another not go/analysis linter required types for deps.
		// Try load cached facts for it.

		for _, act := range lp.actions {
			if !act.loadCachedFacts() {
				// Cached facts loading failed: analyze later the action from source.
				act.needAnalyzeSource = true
				factsCacheDebugf("Loading of facts for already loaded %s failed, analyze it from source later", act)
				act.markDepsForAnalyzingSource()
			}
		}
		return nil
	}

	if lp.isInitial {
		// No need to load cached facts: the package will be analyzed from source
		// because it's the initial.
		return lp.loadFromSource(loadMode)
	}

	return lp.loadImportedPackageWithFacts(loadMode)
}

func (lp *loadingPackage) loadImportedPackageWithFacts(loadMode LoadMode) error {
	pkg := lp.pkg

	// Load package from export data
	if loadMode >= LoadModeTypesInfo {
		if err := lp.loadFromExportData(); err != nil {
			// We asked Go to give us up-to-date export data, yet
			// we can't load it. There must be something wrong.
			//
			// Attempt loading from source. This should fail (because
			// otherwise there would be export data); we just want to
			// get the compile errors. If loading from source succeeds
			// we discard the result, anyway. Otherwise, we'll fail
			// when trying to reload from export data later.

			// Otherwise, it panics because uses already existing (from exported data) types.
			pkg.Types = types.NewPackage(pkg.PkgPath, pkg.Name)
			if srcErr := lp.loadFromSource(loadMode); srcErr != nil {
				return srcErr
			}
			// Make sure this package can't be imported successfully
			pkg.Errors = append(pkg.Errors, packages.Error{
				Pos:  "-",
				Msg:  fmt.Sprintf("could not load export data: %s", err),
				Kind: packages.ParseError,
			})
			return errors.Wrap(err, "could not load export data")
		}
	}

	needLoadFromSource := false
	for _, act := range lp.actions {
		if act.loadCachedFacts() {
			continue
		}

		// Cached facts loading failed: analyze later the action from source.
		factsCacheDebugf("Loading of facts for %s failed, analyze it from source later", act)
		act.needAnalyzeSource = true // can't be set in parallel
		needLoadFromSource = true

		act.markDepsForAnalyzingSource()
	}

	if needLoadFromSource {
		// Cached facts loading failed: analyze later the action from source. To perform
		// the analysis we need to load the package from source code.

		// Otherwise, it panics because uses already existing (from exported data) types.
		if loadMode >= LoadModeTypesInfo {
			pkg.Types = types.NewPackage(pkg.PkgPath, pkg.Name)
		}
		return lp.loadFromSource(loadMode)
	}

	return nil
}

func (lp *loadingPackage) decUse(canClearTypes bool) {
	lp.decUseMutex.Lock()
	defer lp.decUseMutex.Unlock()

	for _, act := range lp.actions {
		pass := act.pass
		if pass == nil {
			continue
		}

		pass.Files = nil
		pass.TypesInfo = nil
		pass.TypesSizes = nil
		pass.ResultOf = nil
		pass.Pkg = nil
		pass.OtherFiles = nil
		pass.AllObjectFacts = nil
		pass.AllPackageFacts = nil
		pass.ImportObjectFact = nil
		pass.ExportObjectFact = nil
		pass.ImportPackageFact = nil
		pass.ExportPackageFact = nil
		act.pass = nil
		act.deps = nil
		if act.result != nil {
			if isMemoryDebug {
				debugf("%s: decUse: nilling act result of size %d bytes", act, sizeOfValueTreeBytes(act.result))
			}
			act.result = nil
		}
	}

	lp.pkg.Syntax = nil
	lp.pkg.TypesInfo = nil
	lp.pkg.TypesSizes = nil

	// Can't set lp.pkg.Imports to nil because of loadFromExportData.visit.

	dependents := atomic.AddInt32(&lp.dependents, -1)
	if dependents != 0 {
		return
	}

	if canClearTypes {
		// canClearTypes is set to true if we can discard type
		// information after the package and its dependents have been
		// processed. This is the case when no whole program checkers (unused) are
		// being run.
		lp.pkg.Types = nil
	}
	lp.pkg = nil

	for _, imp := range lp.imports {
		imp.decUse(canClearTypes)
	}
	lp.imports = nil

	for _, act := range lp.actions {
		if !lp.isInitial {
			act.pkg = nil
		}
		act.packageFacts = nil
		act.objectFacts = nil
	}
	lp.actions = nil
}

func (lp *loadingPackage) convertError(err error) []packages.Error {
	var errs []packages.Error
	// taken from go/packages
	switch err := err.(type) {
	case packages.Error:
		// from driver
		errs = append(errs, err)

	case *os.PathError:
		// from parser
		errs = append(errs, packages.Error{
			Pos:  err.Path + ":1",
			Msg:  err.Err.Error(),
			Kind: packages.ParseError,
		})

	case scanner.ErrorList:
		// from parser
		for _, err := range err {
			errs = append(errs, packages.Error{
				Pos:  err.Pos.String(),
				Msg:  err.Msg,
				Kind: packages.ParseError,
			})
		}

	case types.Error:
		// from type checker
		errs = append(errs, packages.Error{
			Pos:  err.Fset.Position(err.Pos).String(),
			Msg:  err.Msg,
			Kind: packages.TypeError,
		})

	default:
		// unexpected impoverished error from parser?
		errs = append(errs, packages.Error{
			Pos:  "-",
			Msg:  err.Error(),
			Kind: packages.UnknownError,
		})

		// If you see this error message, please file a bug.
		lp.log.Warnf("Internal error: error %q (%T) without position", err, err)
	}
	return errs
}

func (lp *loadingPackage) String() string {
	return fmt.Sprintf("%s@%s", lp.pkg.PkgPath, lp.pkg.Name)
}

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }

func sizeOfValueTreeBytes(v interface{}) int {
	return sizeOfReflectValueTreeBytes(reflect.ValueOf(v), map[uintptr]struct{}{})
}

func sizeOfReflectValueTreeBytes(rv reflect.Value, visitedPtrs map[uintptr]struct{}) int {
	switch rv.Kind() {
	case reflect.Ptr:
		ptrSize := int(rv.Type().Size())
		if rv.IsNil() {
			return ptrSize
		}
		ptr := rv.Pointer()
		if _, ok := visitedPtrs[ptr]; ok {
			return 0
		}
		visitedPtrs[ptr] = struct{}{}
		return ptrSize + sizeOfReflectValueTreeBytes(rv.Elem(), visitedPtrs)
	case reflect.Interface:
		if rv.IsNil() {
			return 0
		}
		return sizeOfReflectValueTreeBytes(rv.Elem(), visitedPtrs)
	case reflect.Struct:
		ret := 0
		for i := 0; i < rv.NumField(); i++ {
			ret += sizeOfReflectValueTreeBytes(rv.Field(i), visitedPtrs)
		}
		return ret
	case reflect.Slice, reflect.Array, reflect.Chan:
		return int(rv.Type().Size()) + rv.Cap()*int(rv.Type().Elem().Size())
	case reflect.Map:
		ret := 0
		for _, key := range rv.MapKeys() {
			mv := rv.MapIndex(key)
			ret += sizeOfReflectValueTreeBytes(key, visitedPtrs)
			ret += sizeOfReflectValueTreeBytes(mv, visitedPtrs)
		}
		return ret
	case reflect.String:
		return rv.Len()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Uintptr, reflect.Bool, reflect.Float32, reflect.Float64,
		reflect.Complex64, reflect.Complex128, reflect.Func, reflect.UnsafePointer:
		return int(rv.Type().Size())
	case reflect.Invalid:
		return 0
	default:
		panic("unknown rv of type " + rv.String())
	}
}
