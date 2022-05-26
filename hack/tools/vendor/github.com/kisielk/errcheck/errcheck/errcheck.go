// Package errcheck is the library used to implement the errcheck command-line tool.
package errcheck

import (
	"bufio"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
)

var errorType *types.Interface

func init() {
	errorType = types.Universe.Lookup("error").Type().Underlying().(*types.Interface)
}

var (
	// ErrNoGoFiles is returned when CheckPackage is run on a package with no Go source files
	ErrNoGoFiles = errors.New("package contains no go source files")

	// DefaultExcludedSymbols is a list of symbol names that are usually excluded from checks by default.
	//
	// Note, that they still need to be explicitly copied to Checker.Exclusions.Symbols
	DefaultExcludedSymbols = []string{
		// bytes
		"(*bytes.Buffer).Write",
		"(*bytes.Buffer).WriteByte",
		"(*bytes.Buffer).WriteRune",
		"(*bytes.Buffer).WriteString",

		// fmt
		"fmt.Errorf",
		"fmt.Print",
		"fmt.Printf",
		"fmt.Println",
		"fmt.Fprint(*bytes.Buffer)",
		"fmt.Fprintf(*bytes.Buffer)",
		"fmt.Fprintln(*bytes.Buffer)",
		"fmt.Fprint(*strings.Builder)",
		"fmt.Fprintf(*strings.Builder)",
		"fmt.Fprintln(*strings.Builder)",
		"fmt.Fprint(os.Stderr)",
		"fmt.Fprintf(os.Stderr)",
		"fmt.Fprintln(os.Stderr)",

		// math/rand
		"math/rand.Read",
		"(*math/rand.Rand).Read",

		// strings
		"(*strings.Builder).Write",
		"(*strings.Builder).WriteByte",
		"(*strings.Builder).WriteRune",
		"(*strings.Builder).WriteString",

		// hash
		"(hash.Hash).Write",
	}
)

// UncheckedError indicates the position of an unchecked error return.
type UncheckedError struct {
	Pos          token.Position
	Line         string
	FuncName     string
	SelectorName string
}

// Result is returned from the CheckPackage function, and holds all the errors
// that were found to be unchecked in a package.
//
// Aggregation can be done using the Append method for users that want to
// combine results from multiple packages.
type Result struct {
	// UncheckedErrors is a list of all the unchecked errors in the package.
	// Printing an error reports its position within the file and the contents of the line.
	UncheckedErrors []UncheckedError
}

type byName []UncheckedError

// Less reports whether the element with index i should sort before the element with index j.
func (b byName) Less(i, j int) bool {
	ei, ej := b[i], b[j]

	pi, pj := ei.Pos, ej.Pos

	if pi.Filename != pj.Filename {
		return pi.Filename < pj.Filename
	}
	if pi.Line != pj.Line {
		return pi.Line < pj.Line
	}
	if pi.Column != pj.Column {
		return pi.Column < pj.Column
	}

	return ei.Line < ej.Line
}

func (b byName) Swap(i, j int) {
	b[i], b[j] = b[j], b[i]
}

func (b byName) Len() int {
	return len(b)
}

// Append appends errors to e. Append does not do any duplicate checking.
func (r *Result) Append(other Result) {
	r.UncheckedErrors = append(r.UncheckedErrors, other.UncheckedErrors...)
}

// Returns the unique errors that have been accumulated. Duplicates may occur
// when a file containing an unchecked error belongs to > 1 package.
//
// The method receiver remains unmodified after the call to Unique.
func (r Result) Unique() Result {
	result := make([]UncheckedError, len(r.UncheckedErrors))
	copy(result, r.UncheckedErrors)
	sort.Sort((byName)(result))
	uniq := result[:0] // compact in-place
	for i, err := range result {
		if i == 0 || err != result[i-1] {
			uniq = append(uniq, err)
		}
	}
	return Result{UncheckedErrors: uniq}
}

// Exclusions define symbols and language elements that will be not checked
type Exclusions struct {

	// Packages lists paths of excluded packages.
	Packages []string

	// SymbolRegexpsByPackage maps individual package paths to regular
	// expressions that match symbols to be excluded.
	//
	// Packages whose paths appear both here and in Packages list will
	// be excluded entirely.
	//
	// This is a legacy input that will be deprecated in errcheck version 2 and
	// should not be used.
	SymbolRegexpsByPackage map[string]*regexp.Regexp

	// Symbols lists patterns that exclude individual package symbols.
	//
	// For example:
	//
	//   "fmt.Errorf"              // function
	//   "fmt.Fprintf(os.Stderr)"  // function with set argument value
	//   "(hash.Hash).Write"       // method
	//
	Symbols []string

	// TestFiles excludes _test.go files.
	TestFiles bool

	// GeneratedFiles excludes generated source files.
	//
	// Source file is assumed to be generated if its contents
	// match the following regular expression:
	//
	//   ^// Code generated .* DO NOT EDIT\\.$
	//
	GeneratedFiles bool

	// BlankAssignments ignores assignments to blank identifier.
	BlankAssignments bool

	// TypeAssertions ignores unchecked type assertions.
	TypeAssertions bool
}

// Checker checks that you checked errors.
type Checker struct {
	// Exclusions defines code packages, symbols, and other elements that will not be checked.
	Exclusions Exclusions

	// Tags are a list of build tags to use.
	Tags []string

	// The mod flag for go build.
	Mod string
}

// loadPackages is used for testing.
var loadPackages = func(cfg *packages.Config, paths ...string) ([]*packages.Package, error) {
	return packages.Load(cfg, paths...)
}

// LoadPackages loads all the packages in all the paths provided. It uses the
// exclusions and build tags provided to by the user when loading the packages.
func (c *Checker) LoadPackages(paths ...string) ([]*packages.Package, error) {
	buildFlags := []string{fmtTags(c.Tags)}
	if c.Mod != "" {
		buildFlags = append(buildFlags, fmt.Sprintf("-mod=%s", c.Mod))
	}
	cfg := &packages.Config{
		Mode:       packages.LoadAllSyntax,
		Tests:      !c.Exclusions.TestFiles,
		BuildFlags: buildFlags,
	}
	return loadPackages(cfg, paths...)
}

var generatedCodeRegexp = regexp.MustCompile("^// Code generated .* DO NOT EDIT\\.$")
var dotStar = regexp.MustCompile(".*")

func (c *Checker) shouldSkipFile(file *ast.File) bool {
	if !c.Exclusions.GeneratedFiles {
		return false
	}

	for _, cg := range file.Comments {
		for _, comment := range cg.List {
			if generatedCodeRegexp.MatchString(comment.Text) {
				return true
			}
		}
	}

	return false
}

// CheckPackage checks packages for errors that have not been checked.
//
// It will exclude specific errors from analysis if the user has configured
// exclusions.
func (c *Checker) CheckPackage(pkg *packages.Package) Result {
	excludedSymbols := map[string]bool{}
	for _, sym := range c.Exclusions.Symbols {
		excludedSymbols[sym] = true
	}

	ignore := map[string]*regexp.Regexp{}
	// Apply SymbolRegexpsByPackage first so that if the same path appears in
	// Packages, a more narrow regexp will be superceded by dotStar below.
	if regexps := c.Exclusions.SymbolRegexpsByPackage; regexps != nil {
		for pkg, re := range regexps {
			// TODO warn if previous entry overwritten?
			ignore[nonVendoredPkgPath(pkg)] = re
		}
	}
	for _, pkg := range c.Exclusions.Packages {
		// TODO warn if previous entry overwritten?
		ignore[nonVendoredPkgPath(pkg)] = dotStar
	}

	v := &visitor{
		pkg:     pkg,
		ignore:  ignore,
		blank:   !c.Exclusions.BlankAssignments,
		asserts: !c.Exclusions.TypeAssertions,
		lines:   make(map[string][]string),
		exclude: excludedSymbols,
		errors:  []UncheckedError{},
	}

	for _, astFile := range v.pkg.Syntax {
		if c.shouldSkipFile(astFile) {
			continue
		}
		ast.Walk(v, astFile)
	}
	return Result{UncheckedErrors: v.errors}
}

// visitor implements the errcheck algorithm
type visitor struct {
	pkg     *packages.Package
	ignore  map[string]*regexp.Regexp
	blank   bool
	asserts bool
	lines   map[string][]string
	exclude map[string]bool

	errors []UncheckedError
}

// selectorAndFunc tries to get the selector and function from call expression.
// For example, given the call expression representing "a.b()", the selector
// is "a.b" and the function is "b" itself.
//
// The final return value will be true if it is able to do extract a selector
// from the call and look up the function object it refers to.
//
// If the call does not include a selector (like if it is a plain "f()" function call)
// then the final return value will be false.
func (v *visitor) selectorAndFunc(call *ast.CallExpr) (*ast.SelectorExpr, *types.Func, bool) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return nil, nil, false
	}

	fn, ok := v.pkg.TypesInfo.ObjectOf(sel.Sel).(*types.Func)
	if !ok {
		// Shouldn't happen, but be paranoid
		return nil, nil, false
	}

	return sel, fn, true

}

// fullName will return a package / receiver-type qualified name for a called function
// if the function is the result of a selector. Otherwise it will return
// the empty string.
//
// The name is fully qualified by the import path, possible type,
// function/method name and pointer receiver.
//
// For example,
//   - for "fmt.Printf(...)" it will return "fmt.Printf"
//   - for "base64.StdEncoding.Decode(...)" it will return "(*encoding/base64.Encoding).Decode"
//   - for "myFunc()" it will return ""
func (v *visitor) fullName(call *ast.CallExpr) string {
	_, fn, ok := v.selectorAndFunc(call)
	if !ok {
		return ""
	}

	// TODO(dh): vendored packages will have /vendor/ in their name,
	// thus not matching vendored standard library packages. If we
	// want to support vendored stdlib packages, we need to implement
	// FullName with our own logic.
	return fn.FullName()
}

func getSelectorName(sel *ast.SelectorExpr) string {
	if ident, ok := sel.X.(*ast.Ident); ok {
		return fmt.Sprintf("%s.%s", ident.Name, sel.Sel.Name)
	}
	if s, ok := sel.X.(*ast.SelectorExpr); ok {
		return fmt.Sprintf("%s.%s", getSelectorName(s), sel.Sel.Name)
	}

	return ""
}

// selectorName will return a name for a called function
// if the function is the result of a selector. Otherwise it will return
// the empty string.
//
// The name is fully qualified by the import path, possible type,
// function/method name and pointer receiver.
//
// For example,
//   - for "fmt.Printf(...)" it will return "fmt.Printf"
//   - for "base64.StdEncoding.Decode(...)" it will return "base64.StdEncoding.Decode"
//   - for "myFunc()" it will return ""
func (v *visitor) selectorName(call *ast.CallExpr) string {
	sel, _, ok := v.selectorAndFunc(call)
	if !ok {
		return ""
	}

	return getSelectorName(sel)
}

// namesForExcludeCheck will return a list of fully-qualified function names
// from a function call that can be used to check against the exclusion list.
//
// If a function call is against a local function (like "myFunc()") then no
// names are returned. If the function is package-qualified (like "fmt.Printf()")
// then just that function's fullName is returned.
//
// Otherwise, we walk through all the potentially embeddded interfaces of the receiver
// the collect a list of type-qualified function names that we will check.
func (v *visitor) namesForExcludeCheck(call *ast.CallExpr) []string {
	sel, fn, ok := v.selectorAndFunc(call)
	if !ok {
		return nil
	}

	name := v.fullName(call)
	if name == "" {
		return nil
	}

	// This will be missing for functions without a receiver (like fmt.Printf),
	// so just fall back to the the function's fullName in that case.
	selection, ok := v.pkg.TypesInfo.Selections[sel]
	if !ok {
		return []string{name}
	}

	// This will return with ok false if the function isn't defined
	// on an interface, so just fall back to the fullName.
	ts, ok := walkThroughEmbeddedInterfaces(selection)
	if !ok {
		return []string{name}
	}

	result := make([]string, len(ts))
	for i, t := range ts {
		// Like in fullName, vendored packages will have /vendor/ in their name,
		// thus not matching vendored standard library packages. If we
		// want to support vendored stdlib packages, we need to implement
		// additional logic here.
		result[i] = fmt.Sprintf("(%s).%s", t.String(), fn.Name())
	}
	return result
}

// isBufferType checks if the expression type is a known in-memory buffer type.
func (v *visitor) argName(expr ast.Expr) string {
	// Special-case literal "os.Stdout" and "os.Stderr"
	if sel, ok := expr.(*ast.SelectorExpr); ok {
		if obj := v.pkg.TypesInfo.ObjectOf(sel.Sel); obj != nil {
			vr, ok := obj.(*types.Var)
			if ok && vr.Pkg() != nil && vr.Pkg().Name() == "os" && (vr.Name() == "Stderr" || vr.Name() == "Stdout") {
				return "os." + vr.Name()
			}
		}
	}
	t := v.pkg.TypesInfo.TypeOf(expr)
	if t == nil {
		return ""
	}
	return t.String()
}

func (v *visitor) excludeCall(call *ast.CallExpr) bool {
	var arg0 string
	if len(call.Args) > 0 {
		arg0 = v.argName(call.Args[0])
	}
	for _, name := range v.namesForExcludeCheck(call) {
		if v.exclude[name] {
			return true
		}
		if arg0 != "" && v.exclude[name+"("+arg0+")"] {
			return true
		}
	}
	return false
}

func (v *visitor) ignoreCall(call *ast.CallExpr) bool {
	if v.excludeCall(call) {
		return true
	}

	// Try to get an identifier.
	// Currently only supports simple expressions:
	//     1. f()
	//     2. x.y.f()
	var id *ast.Ident
	switch exp := call.Fun.(type) {
	case (*ast.Ident):
		id = exp
	case (*ast.SelectorExpr):
		id = exp.Sel
	default:
		// eg: *ast.SliceExpr, *ast.IndexExpr
	}

	if id == nil {
		return false
	}

	// If we got an identifier for the function, see if it is ignored
	if re, ok := v.ignore[""]; ok && re.MatchString(id.Name) {
		return true
	}

	if obj := v.pkg.TypesInfo.Uses[id]; obj != nil {
		if pkg := obj.Pkg(); pkg != nil {
			if re, ok := v.ignore[nonVendoredPkgPath(pkg.Path())]; ok {
				return re.MatchString(id.Name)
			}
		}
	}

	return false
}

// nonVendoredPkgPath returns the unvendored version of the provided package
// path (or returns the provided path if it does not represent a vendored
// path).
func nonVendoredPkgPath(pkgPath string) string {
	lastVendorIndex := strings.LastIndex(pkgPath, "/vendor/")
	if lastVendorIndex == -1 {
		return pkgPath
	}
	return pkgPath[lastVendorIndex+len("/vendor/"):]
}

// errorsByArg returns a slice s such that
// len(s) == number of return types of call
// s[i] == true iff return type at position i from left is an error type
func (v *visitor) errorsByArg(call *ast.CallExpr) []bool {
	switch t := v.pkg.TypesInfo.Types[call].Type.(type) {
	case *types.Named:
		// Single return
		return []bool{isErrorType(t)}
	case *types.Pointer:
		// Single return via pointer
		return []bool{isErrorType(t)}
	case *types.Tuple:
		// Multiple returns
		s := make([]bool, t.Len())
		for i := 0; i < t.Len(); i++ {
			switch et := t.At(i).Type().(type) {
			case *types.Named:
				// Single return
				s[i] = isErrorType(et)
			case *types.Pointer:
				// Single return via pointer
				s[i] = isErrorType(et)
			default:
				s[i] = false
			}
		}
		return s
	}
	return []bool{false}
}

func (v *visitor) callReturnsError(call *ast.CallExpr) bool {
	if v.isRecover(call) {
		return true
	}
	for _, isError := range v.errorsByArg(call) {
		if isError {
			return true
		}
	}
	return false
}

// isRecover returns true if the given CallExpr is a call to the built-in recover() function.
func (v *visitor) isRecover(call *ast.CallExpr) bool {
	if fun, ok := call.Fun.(*ast.Ident); ok {
		if _, ok := v.pkg.TypesInfo.Uses[fun].(*types.Builtin); ok {
			return fun.Name == "recover"
		}
	}
	return false
}

func (v *visitor) addErrorAtPosition(position token.Pos, call *ast.CallExpr) {
	pos := v.pkg.Fset.Position(position)
	lines, ok := v.lines[pos.Filename]
	if !ok {
		lines = readfile(pos.Filename)
		v.lines[pos.Filename] = lines
	}

	line := "??"
	if pos.Line-1 < len(lines) {
		line = strings.TrimSpace(lines[pos.Line-1])
	}

	var name string
	var sel string
	if call != nil {
		name = v.fullName(call)
		sel = v.selectorName(call)
	}

	v.errors = append(v.errors, UncheckedError{pos, line, name, sel})
}

func readfile(filename string) []string {
	var f, err = os.Open(filename)
	if err != nil {
		return nil
	}

	var lines []string
	var scanner = bufio.NewScanner(f)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines
}

func (v *visitor) Visit(node ast.Node) ast.Visitor {
	switch stmt := node.(type) {
	case *ast.ExprStmt:
		if call, ok := stmt.X.(*ast.CallExpr); ok {
			if !v.ignoreCall(call) && v.callReturnsError(call) {
				v.addErrorAtPosition(call.Lparen, call)
			}
		}
	case *ast.GoStmt:
		if !v.ignoreCall(stmt.Call) && v.callReturnsError(stmt.Call) {
			v.addErrorAtPosition(stmt.Call.Lparen, stmt.Call)
		}
	case *ast.DeferStmt:
		if !v.ignoreCall(stmt.Call) && v.callReturnsError(stmt.Call) {
			v.addErrorAtPosition(stmt.Call.Lparen, stmt.Call)
		}
	case *ast.AssignStmt:
		if len(stmt.Rhs) == 1 {
			// single value on rhs; check against lhs identifiers
			if call, ok := stmt.Rhs[0].(*ast.CallExpr); ok {
				if !v.blank {
					break
				}
				if v.ignoreCall(call) {
					break
				}
				isError := v.errorsByArg(call)
				for i := 0; i < len(stmt.Lhs); i++ {
					if id, ok := stmt.Lhs[i].(*ast.Ident); ok {
						// We shortcut calls to recover() because errorsByArg can't
						// check its return types for errors since it returns interface{}.
						if id.Name == "_" && (v.isRecover(call) || isError[i]) {
							v.addErrorAtPosition(id.NamePos, call)
						}
					}
				}
			} else if assert, ok := stmt.Rhs[0].(*ast.TypeAssertExpr); ok {
				if !v.asserts {
					break
				}
				if assert.Type == nil {
					// type switch
					break
				}
				if len(stmt.Lhs) < 2 {
					// assertion result not read
					v.addErrorAtPosition(stmt.Rhs[0].Pos(), nil)
				} else if id, ok := stmt.Lhs[1].(*ast.Ident); ok && v.blank && id.Name == "_" {
					// assertion result ignored
					v.addErrorAtPosition(id.NamePos, nil)
				}
			}
		} else {
			// multiple value on rhs; in this case a call can't return
			// multiple values. Assume len(stmt.Lhs) == len(stmt.Rhs)
			for i := 0; i < len(stmt.Lhs); i++ {
				if id, ok := stmt.Lhs[i].(*ast.Ident); ok {
					if call, ok := stmt.Rhs[i].(*ast.CallExpr); ok {
						if !v.blank {
							continue
						}
						if v.ignoreCall(call) {
							continue
						}
						if id.Name == "_" && v.callReturnsError(call) {
							v.addErrorAtPosition(id.NamePos, call)
						}
					} else if assert, ok := stmt.Rhs[i].(*ast.TypeAssertExpr); ok {
						if !v.asserts {
							continue
						}
						if assert.Type == nil {
							// Shouldn't happen anyway, no multi assignment in type switches
							continue
						}
						v.addErrorAtPosition(id.NamePos, nil)
					}
				}
			}
		}
	default:
	}
	return v
}

func isErrorType(t types.Type) bool {
	return types.Implements(t, errorType)
}
