package durationcheck

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"log"
	"os"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

var Analyzer = &analysis.Analyzer{
	Name:     "durationcheck",
	Doc:      "check for two durations multiplied together",
	Run:      run,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
}

func run(pass *analysis.Pass) (interface{}, error) {
	// if the package does not import time, it can be skipped from analysis
	if !hasImport(pass.Pkg, "time") {
		return nil, nil
	}

	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeTypes := []ast.Node{
		(*ast.BinaryExpr)(nil),
	}

	inspect.Preorder(nodeTypes, check(pass))

	return nil, nil
}

func hasImport(pkg *types.Package, importPath string) bool {
	for _, imp := range pkg.Imports() {
		if imp.Path() == importPath {
			return true
		}
	}

	return false
}

// check contains the logic for checking that time.Duration is used correctly in the code being analysed
func check(pass *analysis.Pass) func(ast.Node) {
	return func(node ast.Node) {
		expr := node.(*ast.BinaryExpr)
		// we are only interested in multiplication
		if expr.Op != token.MUL {
			return
		}

		// get the types of the two operands
		x, xOK := pass.TypesInfo.Types[expr.X]
		y, yOK := pass.TypesInfo.Types[expr.Y]

		if !xOK || !yOK {
			return
		}

		if isDuration(x.Type) && isDuration(y.Type) {
			// check that both sides are acceptable expressions
			if isUnacceptableExpr(pass, expr.X) && isUnacceptableExpr(pass, expr.Y) {
				pass.Reportf(expr.Pos(), "Multiplication of durations: `%s`", formatNode(expr))
			}
		}
	}
}

func isDuration(x types.Type) bool {
	return x.String() == "time.Duration" || x.String() == "*time.Duration"
}

// isUnacceptableExpr returns true if the argument is not an acceptable time.Duration expression
func isUnacceptableExpr(pass *analysis.Pass, expr ast.Expr) bool {
	switch e := expr.(type) {
	case *ast.BasicLit:
		return false
	case *ast.Ident:
		return !isAcceptableNestedExpr(pass, e)
	case *ast.CallExpr:
		return !isAcceptableCast(pass, e)
	case *ast.BinaryExpr:
		return !isAcceptableNestedExpr(pass, e)
	case *ast.UnaryExpr:
		return !isAcceptableNestedExpr(pass, e)
	case *ast.SelectorExpr:
		return !isAcceptableNestedExpr(pass, e)
	case *ast.StarExpr:
		return !isAcceptableNestedExpr(pass, e)
	case *ast.ParenExpr:
		return !isAcceptableNestedExpr(pass, e)
	case *ast.IndexExpr:
		return !isAcceptableNestedExpr(pass, e)
	default:
		return true
	}
}

// isAcceptableCast returns true if the argument is an acceptable expression cast to time.Duration
func isAcceptableCast(pass *analysis.Pass, e *ast.CallExpr) bool {
	// check that there's a single argument
	if len(e.Args) != 1 {
		return false
	}

	// check that the argument is acceptable
	if !isAcceptableNestedExpr(pass, e.Args[0]) {
		return false
	}

	// check for time.Duration cast
	selector, ok := e.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}

	return isDurationCast(selector)
}

func isDurationCast(selector *ast.SelectorExpr) bool {
	pkg, ok := selector.X.(*ast.Ident)
	if !ok {
		return false
	}

	if pkg.Name != "time" {
		return false
	}

	return selector.Sel.Name == "Duration"
}

func isAcceptableNestedExpr(pass *analysis.Pass, n ast.Expr) bool {
	switch e := n.(type) {
	case *ast.BasicLit:
		return true
	case *ast.BinaryExpr:
		return isAcceptableNestedExpr(pass, e.X) && isAcceptableNestedExpr(pass, e.Y)
	case *ast.UnaryExpr:
		return isAcceptableNestedExpr(pass, e.X)
	case *ast.Ident:
		return isAcceptableIdent(pass, e)
	case *ast.CallExpr:
		if isAcceptableCast(pass, e) {
			return true
		}
		t := pass.TypesInfo.TypeOf(e)
		return !isDuration(t)
	case *ast.SelectorExpr:
		return isAcceptableNestedExpr(pass, e.X) && isAcceptableIdent(pass, e.Sel)
	case *ast.StarExpr:
		return isAcceptableNestedExpr(pass, e.X)
	case *ast.ParenExpr:
		return isAcceptableNestedExpr(pass, e.X)
	case *ast.IndexExpr:
		t := pass.TypesInfo.TypeOf(e)
		return !isDuration(t)
	default:
		return false
	}
}

func isAcceptableIdent(pass *analysis.Pass, ident *ast.Ident) bool {
	obj := pass.TypesInfo.ObjectOf(ident)
	return !isDuration(obj.Type())
}

func formatNode(node ast.Node) string {
	buf := new(bytes.Buffer)
	if err := format.Node(buf, token.NewFileSet(), node); err != nil {
		log.Printf("Error formatting expression: %v", err)
		return ""
	}

	return buf.String()
}

func printAST(msg string, node ast.Node) {
	fmt.Printf(">>> %s:\n%s\n\n\n", msg, formatNode(node))
	ast.Fprint(os.Stdout, nil, node, nil)
	fmt.Println("--------------")
}
