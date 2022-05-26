package slowest

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"strings"

	"golang.org/x/tools/go/packages"
	"gotest.tools/gotestsum/log"
	"gotest.tools/gotestsum/testjson"
)

func writeTestSkip(tcs []testjson.TestCase, skipStmt ast.Stmt) error {
	fset := token.NewFileSet()
	cfg := packages.Config{
		Mode:       modeAll(),
		Tests:      true,
		Fset:       fset,
		BuildFlags: buildFlags(),
	}
	pkgNames, index := testNamesByPkgName(tcs)
	pkgs, err := packages.Load(&cfg, pkgNames...)
	if err != nil {
		return fmt.Errorf("failed to load packages: %v", err)
	}

	for _, pkg := range pkgs {
		if len(pkg.Errors) > 0 {
			return errPkgLoad(pkg)
		}
		tcs, ok := index[normalizePkgName(pkg.PkgPath)]
		if !ok {
			log.Debugf("skipping %v, no slow tests", pkg.PkgPath)
			continue
		}

		log.Debugf("rewriting %v for %d test cases", pkg.PkgPath, len(tcs))
		for _, file := range pkg.Syntax {
			path := fset.File(file.Pos()).Name()
			log.Debugf("looking for test cases in: %v", path)
			if !rewriteAST(file, tcs, skipStmt) {
				continue
			}
			if err := writeFile(path, file, fset); err != nil {
				return fmt.Errorf("failed to write ast to file %v: %v", path, err)
			}
		}
	}
	return errTestCasesNotFound(index)
}

// normalizePkgName removes the _test suffix from a package name. External test
// packages (those named package_test) may contain tests, but the test2json output
// always uses the non-external package name. The _test suffix must be removed
// so that any slow tests in an external test package can be found.
func normalizePkgName(name string) string {
	return strings.TrimSuffix(name, "_test")
}

func writeFile(path string, file *ast.File, fset *token.FileSet) error {
	fh, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		if err := fh.Close(); err != nil {
			log.Errorf("Failed to close file %v: %v", path, err)
		}
	}()
	return format.Node(fh, fset, file)
}

func parseSkipStatement(text string) (ast.Stmt, error) {
	switch text {
	case "default", "testing.Short":
		text = `
	if testing.Short() {
		t.Skip("too slow for testing.Short")
	}
`
	}
	// Add some required boilerplate around the statement to make it a valid file
	text = "package stub\nfunc Stub() {\n" + text + "\n}\n"
	file, err := parser.ParseFile(token.NewFileSet(), "fragment", text, 0)
	if err != nil {
		return nil, err
	}
	stmt := file.Decls[0].(*ast.FuncDecl).Body.List[0]
	return stmt, nil
}

func rewriteAST(file *ast.File, testNames set, skipStmt ast.Stmt) bool {
	var modified bool
	for _, decl := range file.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		name := fd.Name.Name // TODO: can this be nil?
		if _, ok := testNames[name]; !ok {
			continue
		}

		fd.Body.List = append([]ast.Stmt{skipStmt}, fd.Body.List...)
		modified = true
		delete(testNames, name)
	}
	return modified
}

type set map[string]struct{}

// testNamesByPkgName strips subtest names from test names, then builds
// and returns a slice of all the packages names, and a mapping of package name
// to set of failed tests in that package.
//
// subtests are removed because the AST lookup currently only works for top-level
// functions, not t.Run subtests.
func testNamesByPkgName(tcs []testjson.TestCase) ([]string, map[string]set) {
	var pkgs []string
	index := make(map[string]set)
	for _, tc := range tcs {
		testName := tc.Test.Name()
		if tc.Test.IsSubTest() {
			root, _ := tc.Test.Split()
			testName = root
		}
		if len(index[tc.Package]) == 0 {
			pkgs = append(pkgs, tc.Package)
			index[tc.Package] = make(map[string]struct{})
		}
		index[tc.Package][testName] = struct{}{}
	}
	return pkgs, index
}

func errPkgLoad(pkg *packages.Package) error {
	buf := new(strings.Builder)
	for _, err := range pkg.Errors {
		buf.WriteString("\n" + err.Error())
	}
	return fmt.Errorf("failed to load package %v %v", pkg.PkgPath, buf.String())
}

func errTestCasesNotFound(index map[string]set) error {
	var missed []string
	for pkg, tcs := range index {
		for tc := range tcs {
			missed = append(missed, fmt.Sprintf("%v.%v", pkg, tc))
		}
	}
	if len(missed) == 0 {
		return nil
	}
	return fmt.Errorf("failed to find source for test cases:\n%v", strings.Join(missed, "\n"))
}

func modeAll() packages.LoadMode {
	mode := packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles
	mode = mode | packages.NeedImports | packages.NeedDeps
	mode = mode | packages.NeedTypes | packages.NeedTypesSizes
	mode = mode | packages.NeedSyntax | packages.NeedTypesInfo
	return mode
}

func buildFlags() []string {
	flags := os.Getenv("GOFLAGS")
	if len(flags) == 0 {
		return nil
	}
	return strings.Split(os.Getenv("GOFLAGS"), " ")
}
