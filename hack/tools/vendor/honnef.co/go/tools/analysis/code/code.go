// Package code answers structural and type questions about Go code.
package code

import (
	"flag"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"strings"

	"honnef.co/go/tools/analysis/facts"
	"honnef.co/go/tools/go/ast/astutil"
	"honnef.co/go/tools/go/types/typeutil"

	"golang.org/x/tools/go/analysis"
)

type Positioner interface {
	Pos() token.Pos
}

func IsOfType(pass *analysis.Pass, expr ast.Expr, name string) bool {
	return typeutil.IsType(pass.TypesInfo.TypeOf(expr), name)
}

func IsInTest(pass *analysis.Pass, node Positioner) bool {
	// FIXME(dh): this doesn't work for global variables with
	// initializers
	f := pass.Fset.File(node.Pos())
	return f != nil && strings.HasSuffix(f.Name(), "_test.go")
}

// IsMain reports whether the package being processed is a package
// main.
func IsMain(pass *analysis.Pass) bool {
	return pass.Pkg.Name() == "main"
}

// IsMainLike reports whether the package being processed is a
// main-like package. A main-like package is a package that is
// package main, or that is intended to be used by a tool framework
// such as cobra to implement a command.
//
// Note that this function errs on the side of false positives; it may
// return true for packages that aren't main-like. IsMainLike is
// intended for analyses that wish to suppress diagnostics for
// main-like packages to avoid false positives.
func IsMainLike(pass *analysis.Pass) bool {
	if pass.Pkg.Name() == "main" {
		return true
	}
	for _, imp := range pass.Pkg.Imports() {
		if imp.Path() == "github.com/spf13/cobra" {
			return true
		}
	}
	return false
}

func SelectorName(pass *analysis.Pass, expr *ast.SelectorExpr) string {
	info := pass.TypesInfo
	sel := info.Selections[expr]
	if sel == nil {
		if x, ok := expr.X.(*ast.Ident); ok {
			pkg, ok := info.ObjectOf(x).(*types.PkgName)
			if !ok {
				// This shouldn't happen
				return fmt.Sprintf("%s.%s", x.Name, expr.Sel.Name)
			}
			return fmt.Sprintf("%s.%s", pkg.Imported().Path(), expr.Sel.Name)
		}
		panic(fmt.Sprintf("unsupported selector: %v", expr))
	}
	return fmt.Sprintf("(%s).%s", sel.Recv(), sel.Obj().Name())
}

func IsNil(pass *analysis.Pass, expr ast.Expr) bool {
	return pass.TypesInfo.Types[expr].IsNil()
}

func BoolConst(pass *analysis.Pass, expr ast.Expr) bool {
	val := pass.TypesInfo.ObjectOf(expr.(*ast.Ident)).(*types.Const).Val()
	return constant.BoolVal(val)
}

func IsBoolConst(pass *analysis.Pass, expr ast.Expr) bool {
	// We explicitly don't support typed bools because more often than
	// not, custom bool types are used as binary enums and the
	// explicit comparison is desired.

	ident, ok := expr.(*ast.Ident)
	if !ok {
		return false
	}
	obj := pass.TypesInfo.ObjectOf(ident)
	c, ok := obj.(*types.Const)
	if !ok {
		return false
	}
	basic, ok := c.Type().(*types.Basic)
	if !ok {
		return false
	}
	if basic.Kind() != types.UntypedBool && basic.Kind() != types.Bool {
		return false
	}
	return true
}

func ExprToInt(pass *analysis.Pass, expr ast.Expr) (int64, bool) {
	tv := pass.TypesInfo.Types[expr]
	if tv.Value == nil {
		return 0, false
	}
	if tv.Value.Kind() != constant.Int {
		return 0, false
	}
	return constant.Int64Val(tv.Value)
}

func ExprToString(pass *analysis.Pass, expr ast.Expr) (string, bool) {
	val := pass.TypesInfo.Types[expr].Value
	if val == nil {
		return "", false
	}
	if val.Kind() != constant.String {
		return "", false
	}
	return constant.StringVal(val), true
}

func CallName(pass *analysis.Pass, call *ast.CallExpr) string {
	switch fun := astutil.Unparen(call.Fun).(type) {
	case *ast.SelectorExpr:
		fn, ok := pass.TypesInfo.ObjectOf(fun.Sel).(*types.Func)
		if !ok {
			return ""
		}
		return typeutil.FuncName(fn)
	case *ast.Ident:
		obj := pass.TypesInfo.ObjectOf(fun)
		switch obj := obj.(type) {
		case *types.Func:
			return typeutil.FuncName(obj)
		case *types.Builtin:
			return obj.Name()
		default:
			return ""
		}
	default:
		return ""
	}
}

func IsCallTo(pass *analysis.Pass, node ast.Node, name string) bool {
	call, ok := node.(*ast.CallExpr)
	if !ok {
		return false
	}
	return CallName(pass, call) == name
}

func IsCallToAny(pass *analysis.Pass, node ast.Node, names ...string) bool {
	call, ok := node.(*ast.CallExpr)
	if !ok {
		return false
	}
	q := CallName(pass, call)
	for _, name := range names {
		if q == name {
			return true
		}
	}
	return false
}

func File(pass *analysis.Pass, node Positioner) *ast.File {
	m := pass.ResultOf[facts.TokenFile].(map[*token.File]*ast.File)
	return m[pass.Fset.File(node.Pos())]
}

// IsGenerated reports whether pos is in a generated file, It ignores
// //line directives.
func IsGenerated(pass *analysis.Pass, pos token.Pos) bool {
	_, ok := Generator(pass, pos)
	return ok
}

// Generator returns the generator that generated the file containing
// pos. It ignores //line directives.
func Generator(pass *analysis.Pass, pos token.Pos) (facts.Generator, bool) {
	file := pass.Fset.PositionFor(pos, false).Filename
	m := pass.ResultOf[facts.Generated].(map[string]facts.Generator)
	g, ok := m[file]
	return g, ok
}

// MayHaveSideEffects reports whether expr may have side effects. If
// the purity argument is nil, this function implements a purely
// syntactic check, meaning that any function call may have side
// effects, regardless of the called function's body. Otherwise,
// purity will be consulted to determine the purity of function calls.
func MayHaveSideEffects(pass *analysis.Pass, expr ast.Expr, purity facts.PurityResult) bool {
	switch expr := expr.(type) {
	case *ast.BadExpr:
		return true
	case *ast.Ellipsis:
		return MayHaveSideEffects(pass, expr.Elt, purity)
	case *ast.FuncLit:
		// the literal itself cannot have side effects, only calling it
		// might, which is handled by CallExpr.
		return false
	case *ast.ArrayType, *ast.StructType, *ast.FuncType, *ast.InterfaceType, *ast.MapType, *ast.ChanType:
		// types cannot have side effects
		return false
	case *ast.BasicLit:
		return false
	case *ast.BinaryExpr:
		return MayHaveSideEffects(pass, expr.X, purity) || MayHaveSideEffects(pass, expr.Y, purity)
	case *ast.CallExpr:
		if purity == nil {
			return true
		}
		switch obj := typeutil.Callee(pass.TypesInfo, expr).(type) {
		case *types.Func:
			if _, ok := purity[obj]; !ok {
				return true
			}
		case *types.Builtin:
			switch obj.Name() {
			case "len", "cap":
			default:
				return true
			}
		default:
			return true
		}
		for _, arg := range expr.Args {
			if MayHaveSideEffects(pass, arg, purity) {
				return true
			}
		}
		return false
	case *ast.CompositeLit:
		if MayHaveSideEffects(pass, expr.Type, purity) {
			return true
		}
		for _, elt := range expr.Elts {
			if MayHaveSideEffects(pass, elt, purity) {
				return true
			}
		}
		return false
	case *ast.Ident:
		return false
	case *ast.IndexExpr:
		return MayHaveSideEffects(pass, expr.X, purity) || MayHaveSideEffects(pass, expr.Index, purity)
	case *ast.KeyValueExpr:
		return MayHaveSideEffects(pass, expr.Key, purity) || MayHaveSideEffects(pass, expr.Value, purity)
	case *ast.SelectorExpr:
		return MayHaveSideEffects(pass, expr.X, purity)
	case *ast.SliceExpr:
		return MayHaveSideEffects(pass, expr.X, purity) ||
			MayHaveSideEffects(pass, expr.Low, purity) ||
			MayHaveSideEffects(pass, expr.High, purity) ||
			MayHaveSideEffects(pass, expr.Max, purity)
	case *ast.StarExpr:
		return MayHaveSideEffects(pass, expr.X, purity)
	case *ast.TypeAssertExpr:
		return MayHaveSideEffects(pass, expr.X, purity)
	case *ast.UnaryExpr:
		if MayHaveSideEffects(pass, expr.X, purity) {
			return true
		}
		return expr.Op == token.ARROW
	case *ast.ParenExpr:
		return MayHaveSideEffects(pass, expr.X, purity)
	case nil:
		return false
	default:
		panic(fmt.Sprintf("internal error: unhandled type %T", expr))
	}
}

func IsGoVersion(pass *analysis.Pass, minor int) bool {
	f, ok := pass.Analyzer.Flags.Lookup("go").Value.(flag.Getter)
	if !ok {
		panic("requested Go version, but analyzer has no version flag")
	}
	version := f.Get().(int)
	return version >= minor
}
