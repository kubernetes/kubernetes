package loader

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"go/types"
	"log"
	"os"

	"golang.org/x/tools/go/gcexportdata"
	"golang.org/x/tools/go/packages"
)

// Graph resolves patterns and returns packages with all the
// information required to later load type information, and optionally
// syntax trees.
//
// The provided config can set any setting with the exception of Mode.
func Graph(cfg packages.Config, patterns ...string) ([]*packages.Package, error) {
	cfg.Mode = packages.NeedName | packages.NeedImports | packages.NeedDeps | packages.NeedExportsFile | packages.NeedFiles | packages.NeedCompiledGoFiles | packages.NeedTypesSizes
	pkgs, err := packages.Load(&cfg, patterns...)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	packages.Visit(pkgs, nil, func(pkg *packages.Package) {
		pkg.Fset = fset
	})

	n := 0
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
		pkgs[n] = pkg
		n++
	}
	return pkgs[:n], nil
}

// LoadFromExport loads a package from export data. All of its
// dependencies must have been loaded already.
func LoadFromExport(pkg *packages.Package) error {
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

// LoadFromSource loads a package from source. All of its dependencies
// must have been loaded already.
func LoadFromSource(pkg *packages.Package) error {
	pkg.IllTyped = true
	pkg.Types = types.NewPackage(pkg.PkgPath, pkg.Name)

	// OPT(dh): many packages have few files, much fewer than there
	// are CPU cores. Additionally, parsing each individual file is
	// very fast. A naive parallel implementation of this loop won't
	// be faster, and tends to be slower due to extra scheduling,
	// bookkeeping and potentially false sharing of cache lines.
	pkg.Syntax = make([]*ast.File, len(pkg.CompiledGoFiles))
	for i, file := range pkg.CompiledGoFiles {
		f, err := parser.ParseFile(pkg.Fset, file, nil, parser.ParseComments)
		if err != nil {
			pkg.Errors = append(pkg.Errors, convertError(err)...)
			return err
		}
		pkg.Syntax[i] = f
	}
	pkg.TypesInfo = &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Scopes:     make(map[ast.Node]*types.Scope),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
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
			pkg.Errors = append(pkg.Errors, convertError(err)...)
		},
	}
	err := types.NewChecker(tc, pkg.Fset, pkg.Types, pkg.TypesInfo).Files(pkg.Syntax)
	if err != nil {
		return err
	}
	pkg.IllTyped = false
	return nil
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
		// unexpected impoverished error from parser?
		errs = append(errs, packages.Error{
			Pos:  "-",
			Msg:  err.Error(),
			Kind: packages.UnknownError,
		})

		// If you see this error message, please file a bug.
		log.Printf("internal error: error %q (%T) without position", err, err)
	}
	return errs
}

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }
