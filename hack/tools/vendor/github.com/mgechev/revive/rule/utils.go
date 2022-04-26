package rule

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"go/types"
	"regexp"
	"strings"

	"github.com/mgechev/revive/lint"
)

const styleGuideBase = "https://golang.org/wiki/CodeReviewComments"

// isBlank returns whether id is the blank identifier "_".
// If id == nil, the answer is false.
func isBlank(id *ast.Ident) bool { return id != nil && id.Name == "_" }

func isTest(f *lint.File) bool {
	return strings.HasSuffix(f.Name, "_test.go")
}

var commonMethods = map[string]bool{
	"Error":     true,
	"Read":      true,
	"ServeHTTP": true,
	"String":    true,
	"Write":     true,
	"Unwrap":    true,
}

func receiverType(fn *ast.FuncDecl) string {
	switch e := fn.Recv.List[0].Type.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.StarExpr:
		if id, ok := e.X.(*ast.Ident); ok {
			return id.Name
		}
	}
	// The parser accepts much more than just the legal forms.
	return "invalid-type"
}

var knownNameExceptions = map[string]bool{
	"LastInsertId": true, // must match database/sql
	"kWh":          true,
}

func isCgoExported(f *ast.FuncDecl) bool {
	if f.Recv != nil || f.Doc == nil {
		return false
	}

	cgoExport := regexp.MustCompile(fmt.Sprintf("(?m)^//export %s$", regexp.QuoteMeta(f.Name.Name)))
	for _, c := range f.Doc.List {
		if cgoExport.MatchString(c.Text) {
			return true
		}
	}
	return false
}

var allCapsRE = regexp.MustCompile(`^[A-Z0-9_]+$`)

func isIdent(expr ast.Expr, ident string) bool {
	id, ok := expr.(*ast.Ident)
	return ok && id.Name == ident
}

var zeroLiteral = map[string]bool{
	"false": true, // bool
	// runes
	`'\x00'`: true,
	`'\000'`: true,
	// strings
	`""`: true,
	"``": true,
	// numerics
	"0":   true,
	"0.":  true,
	"0.0": true,
	"0i":  true,
}

func validType(T types.Type) bool {
	return T != nil &&
		T != types.Typ[types.Invalid] &&
		!strings.Contains(T.String(), "invalid type") // good but not foolproof
}

// isPkgDot checks if the expression is <pkg>.<name>
func isPkgDot(expr ast.Expr, pkg, name string) bool {
	sel, ok := expr.(*ast.SelectorExpr)
	return ok && isIdent(sel.X, pkg) && isIdent(sel.Sel, name)
}

func srcLine(src []byte, p token.Position) string {
	// Run to end of line in both directions if not at line start/end.
	lo, hi := p.Offset, p.Offset+1
	for lo > 0 && src[lo-1] != '\n' {
		lo--
	}
	for hi < len(src) && src[hi-1] != '\n' {
		hi++
	}
	return string(src[lo:hi])
}

// pick yields a list of nodes by picking them from a sub-ast with root node n.
// Nodes are selected by applying the fselect function
// f function is applied to each selected node before inserting it in the final result.
// If f==nil then it defaults to the identity function (ie it returns the node itself)
func pick(n ast.Node, fselect func(n ast.Node) bool, f func(n ast.Node) []ast.Node) []ast.Node {
	var result []ast.Node

	if n == nil {
		return result
	}

	if f == nil {
		f = func(n ast.Node) []ast.Node { return []ast.Node{n} }
	}

	onSelect := func(n ast.Node) {
		result = append(result, f(n)...)
	}
	p := picker{fselect: fselect, onSelect: onSelect}
	ast.Walk(p, n)
	return result
}

type picker struct {
	fselect  func(n ast.Node) bool
	onSelect func(n ast.Node)
}

func (p picker) Visit(node ast.Node) ast.Visitor {
	if p.fselect == nil {
		return nil
	}

	if p.fselect(node) {
		p.onSelect(node)
	}

	return p
}

// isBoolOp returns true if the given token corresponds to
// a bool operator
func isBoolOp(t token.Token) bool {
	switch t {
	case token.LAND, token.LOR, token.EQL, token.NEQ:
		return true
	}

	return false
}

const (
	trueName  = "true"
	falseName = "false"
)

func isExprABooleanLit(n ast.Node) (lexeme string, ok bool) {
	oper, ok := n.(*ast.Ident)

	if !ok {
		return "", false
	}

	return oper.Name, (oper.Name == trueName || oper.Name == falseName)
}

// gofmt returns a string representation of an AST subtree.
func gofmt(x interface{}) string {
	buf := bytes.Buffer{}
	fs := token.NewFileSet()
	printer.Fprint(&buf, fs, x)
	return buf.String()
}

// checkNumberOfArguments fails if the given number of arguments is not, at least, the expected one
func checkNumberOfArguments(expected int, args lint.Arguments, ruleName string) {
	if len(args) < expected {
		panic(fmt.Sprintf("not enough arguments for %s rule, expected %d, got %d. Please check the rule's documentation", ruleName, expected, len(args)))
	}
}
