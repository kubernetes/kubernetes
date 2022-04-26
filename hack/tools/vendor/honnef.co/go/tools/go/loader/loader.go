package loader

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"go/types"
	"os"
	"time"

	"honnef.co/go/tools/config"
	"honnef.co/go/tools/internal/cache"
	"honnef.co/go/tools/internal/go/gcimporter"

	"golang.org/x/tools/go/packages"
)

const MaxFileSize = 50 * 1024 * 1024 // 50 MB

var errMaxFileSize = errors.New("file exceeds max file size")

type PackageSpec struct {
	ID      string
	Name    string
	PkgPath string
	// Errors that occurred while building the import graph. These will
	// primarily be parse errors or failure to resolve imports, but
	// may also be other errors.
	Errors          []packages.Error
	GoFiles         []string
	CompiledGoFiles []string
	OtherFiles      []string
	ExportFile      string
	Imports         map[string]*PackageSpec
	TypesSizes      types.Sizes
	Hash            cache.ActionID
	Module          *packages.Module

	Config config.Config
}

func (spec *PackageSpec) String() string {
	return spec.ID
}

type Package struct {
	*PackageSpec

	// Errors that occurred while loading the package. These will
	// primarily be parse or type errors, but may also be lower-level
	// failures such as file-system ones.
	Errors    []packages.Error
	Types     *types.Package
	Fset      *token.FileSet
	Syntax    []*ast.File
	TypesInfo *types.Info
}

// Graph resolves patterns and returns packages with all the
// information required to later load type information, and optionally
// syntax trees.
//
// The provided config can set any setting with the exception of Mode.
func Graph(c *cache.Cache, cfg *packages.Config, patterns ...string) ([]*PackageSpec, error) {
	var dcfg packages.Config
	if cfg != nil {
		dcfg = *cfg
	}
	dcfg.Mode = packages.NeedName |
		packages.NeedImports |
		packages.NeedDeps |
		packages.NeedExportsFile |
		packages.NeedFiles |
		packages.NeedCompiledGoFiles |
		packages.NeedTypesSizes |
		packages.NeedModule
	pkgs, err := packages.Load(&dcfg, patterns...)
	if err != nil {
		return nil, err
	}

	m := map[*packages.Package]*PackageSpec{}
	packages.Visit(pkgs, nil, func(pkg *packages.Package) {
		spec := &PackageSpec{
			ID:              pkg.ID,
			Name:            pkg.Name,
			PkgPath:         pkg.PkgPath,
			Errors:          pkg.Errors,
			GoFiles:         pkg.GoFiles,
			CompiledGoFiles: pkg.CompiledGoFiles,
			OtherFiles:      pkg.OtherFiles,
			ExportFile:      pkg.ExportFile,
			Imports:         map[string]*PackageSpec{},
			TypesSizes:      pkg.TypesSizes,
			Module:          pkg.Module,
		}
		for path, imp := range pkg.Imports {
			spec.Imports[path] = m[imp]
		}
		if cdir := config.Dir(pkg.GoFiles); cdir != "" {
			cfg, err := config.Load(cdir)
			if err != nil {
				spec.Errors = append(spec.Errors, convertError(err)...)
			}
			spec.Config = cfg
		} else {
			spec.Config = config.DefaultConfig
		}
		spec.Hash, err = computeHash(c, spec)
		if err != nil {
			spec.Errors = append(spec.Errors, convertError(err)...)
		}
		m[pkg] = spec
	})
	out := make([]*PackageSpec, 0, len(pkgs))
	for _, pkg := range pkgs {
		if len(pkg.CompiledGoFiles) == 0 && len(pkg.Errors) == 0 && pkg.PkgPath != "unsafe" {
			// If a package consists only of test files, then
			// go/packages incorrectly(?) returns an empty package for
			// the non-test variant. Get rid of those packages. See
			// #646.
			//
			// Do not, however, skip packages that have errors. Those,
			// too, may have no files, but we want to print the
			// errors.
			continue
		}
		out = append(out, m[pkg])
	}

	return out, nil
}

type program struct {
	fset     *token.FileSet
	packages map[string]*types.Package
}

type Stats struct {
	Source time.Duration
	Export map[*PackageSpec]time.Duration
}

// Load loads the package described in spec. Imports will be loaded
// from export data, while the package itself will be loaded from
// source.
//
// An error will only be returned for system failures, such as failure
// to read export data from disk. Syntax and type errors, among
// others, will only populate the returned package's Errors field.
func Load(spec *PackageSpec) (*Package, Stats, error) {
	prog := &program{
		fset:     token.NewFileSet(),
		packages: map[string]*types.Package{},
	}

	stats := Stats{
		Export: map[*PackageSpec]time.Duration{},
	}
	var b []byte
	for _, imp := range spec.Imports {
		if imp.PkgPath == "unsafe" {
			continue
		}
		t := time.Now()
		var err error
		_, b, err = prog.loadFromExport(imp, b)
		stats.Export[imp] = time.Since(t)
		if err != nil {
			return nil, stats, err
		}
	}
	t := time.Now()
	pkg, err := prog.loadFromSource(spec)
	if err == errMaxFileSize {
		pkg, _, err = prog.loadFromExport(spec, b)
	}
	stats.Source = time.Since(t)
	return pkg, stats, err
}

// loadFromExport loads a package from export data.
func (prog *program) loadFromExport(spec *PackageSpec, b []byte) (*Package, []byte, error) {
	// log.Printf("Loading package %s from export", spec)
	if spec.ExportFile == "" {
		return nil, b, fmt.Errorf("no export data for %q", spec.ID)
	}
	f, err := os.Open(spec.ExportFile)
	if err != nil {
		return nil, b, err
	}
	defer f.Close()

	b, err = gcimporter.GetExportData(f, b)
	if err != nil {
		return nil, b, err
	}

	_, tpkg, err := gcimporter.IImportData(prog.fset, prog.packages, b[1:], spec.PkgPath)
	if err != nil {
		return nil, b, err
	}
	pkg := &Package{
		PackageSpec: spec,
		Types:       tpkg,
		Fset:        prog.fset,
	}
	// runtime.SetFinalizer(pkg, func(pkg *Package) {
	// 	log.Println("Unloading package", pkg.PkgPath)
	// })
	return pkg, b, nil
}

// loadFromSource loads a package from source. All of its dependencies
// must have been loaded already.
func (prog *program) loadFromSource(spec *PackageSpec) (*Package, error) {
	if len(spec.Errors) > 0 {
		panic("LoadFromSource called on package with errors")
	}

	pkg := &Package{
		PackageSpec: spec,
		Types:       types.NewPackage(spec.PkgPath, spec.Name),
		Syntax:      make([]*ast.File, len(spec.CompiledGoFiles)),
		Fset:        prog.fset,
		TypesInfo: &types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Scopes:     make(map[ast.Node]*types.Scope),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
		},
	}
	// runtime.SetFinalizer(pkg, func(pkg *Package) {
	// 	log.Println("Unloading package", pkg.PkgPath)
	// })

	// OPT(dh): many packages have few files, much fewer than there
	// are CPU cores. Additionally, parsing each individual file is
	// very fast. A naive parallel implementation of this loop won't
	// be faster, and tends to be slower due to extra scheduling,
	// bookkeeping and potentially false sharing of cache lines.
	for i, file := range spec.CompiledGoFiles {
		f, err := os.Open(file)
		if err != nil {
			return nil, err
		}
		fi, err := f.Stat()
		if err != nil {
			return nil, err
		}
		if fi.Size() >= MaxFileSize {
			return nil, errMaxFileSize
		}
		af, err := parser.ParseFile(prog.fset, file, f, parser.ParseComments)
		f.Close()
		if err != nil {
			pkg.Errors = append(pkg.Errors, convertError(err)...)
			return pkg, nil
		}
		pkg.Syntax[i] = af
	}
	importer := func(path string) (*types.Package, error) {
		if path == "unsafe" {
			return types.Unsafe, nil
		}
		if path == "C" {
			// go/packages doesn't tell us that cgo preprocessing
			// failed. When we subsequently try to parse the package,
			// we'll encounter the raw C import.
			return nil, errors.New("cgo preprocessing failed")
		}
		ispecpkg := spec.Imports[path]
		if ispecpkg == nil {
			return nil, fmt.Errorf("trying to import %q in the context of %q returned nil PackageSpec", path, spec)
		}
		ipkg := prog.packages[ispecpkg.PkgPath]
		if ipkg == nil {
			return nil, fmt.Errorf("trying to import %q (%q) in the context of %q returned nil PackageSpec", ispecpkg.PkgPath, path, spec)
		}
		return ipkg, nil
	}
	tc := &types.Config{
		Importer: importerFunc(importer),
		Error: func(err error) {
			pkg.Errors = append(pkg.Errors, convertError(err)...)
		},
	}
	types.NewChecker(tc, pkg.Fset, pkg.Types, pkg.TypesInfo).Files(pkg.Syntax)
	return pkg, nil
}

func convertError(err error) []packages.Error {
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
		errs = append(errs, packages.Error{
			Pos:  "-",
			Msg:  err.Error(),
			Kind: packages.UnknownError,
		})
	}
	return errs
}

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }
