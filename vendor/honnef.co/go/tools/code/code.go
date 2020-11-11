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

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/ast/inspector"
	"honnef.co/go/tools/facts"
	"honnef.co/go/tools/go/types/typeutil"
	"honnef.co/go/tools/ir"
	"honnef.co/go/tools/lint"
)

type Positioner interface {
	Pos() token.Pos
}

func CallName(call *ir.CallCommon) string {
	if call.IsInvoke() {
		return ""
	}
	switch v := call.Value.(type) {
	case *ir.Function:
		fn, ok := v.Object().(*types.Func)
		if !ok {
			return ""
		}
		return lint.FuncName(fn)
	case *ir.Builtin:
		return v.Name()
	}
	return ""
}

func IsCallTo(call *ir.CallCommon, name string) bool { return CallName(call) == name }

func IsCallToAny(call *ir.CallCommon, names ...string) bool {
	q := CallName(call)
	for _, name := range names {
		if q == name {
			return true
		}
	}
	return false
}

func IsType(T types.Type, name string) bool { return types.TypeString(T, nil) == name }

func FilterDebug(instr []ir.Instruction) []ir.Instruction {
	var out []ir.Instruction
	for _, ins := range instr {
		if _, ok := ins.(*ir.DebugRef); !ok {
			out = append(out, ins)
		}
	}
	return out
}

func IsExample(fn *ir.Function) bool {
	if !strings.HasPrefix(fn.Name(), "Example") {
		return false
	}
	f := fn.Prog.Fset.File(fn.Pos())
	if f == nil {
		return false
	}
	return strings.HasSuffix(f.Name(), "_test.go")
}

func IsPointerLike(T types.Type) bool {
	switch T := T.Underlying().(type) {
	case *types.Interface, *types.Chan, *types.Map, *types.Signature, *types.Pointer:
		return true
	case *types.Basic:
		return T.Kind() == types.UnsafePointer
	}
	return false
}

func IsIdent(expr ast.Expr, ident string) bool {
	id, ok := expr.(*ast.Ident)
	return ok && id.Name == ident
}

// isBlank returns whether id is the blank identifier "_".
// If id == nil, the answer is false.
func IsBlank(id ast.Expr) bool {
	ident, _ := id.(*ast.Ident)
	return ident != nil && ident.Name == "_"
}

func IsIntLiteral(expr ast.Expr, literal string) bool {
	lit, ok := expr.(*ast.BasicLit)
	return ok && lit.Kind == token.INT && lit.Value == literal
}

// Deprecated: use IsIntLiteral instead
func IsZero(expr ast.Expr) bool {
	return IsIntLiteral(expr, "0")
}

func IsOfType(pass *analysis.Pass, expr ast.Expr, name string) bool {
	return IsType(pass.TypesInfo.TypeOf(expr), name)
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

// Dereference returns a pointer's element type; otherwise it returns
// T.
func Dereference(T types.Type) types.Type {
	if p, ok := T.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return T
}

// DereferenceR returns a pointer's element type; otherwise it returns
// T. If the element type is itself a pointer, DereferenceR will be
// applied recursively.
func DereferenceR(T types.Type) types.Type {
	if p, ok := T.Underlying().(*types.Pointer); ok {
		return DereferenceR(p.Elem())
	}
	return T
}

func CallNameAST(pass *analysis.Pass, call *ast.CallExpr) string {
	switch fun := astutil.Unparen(call.Fun).(type) {
	case *ast.SelectorExpr:
		fn, ok := pass.TypesInfo.ObjectOf(fun.Sel).(*types.Func)
		if !ok {
			return ""
		}
		return lint.FuncName(fn)
	case *ast.Ident:
		obj := pass.TypesInfo.ObjectOf(fun)
		switch obj := obj.(type) {
		case *types.Func:
			return lint.FuncName(obj)
		case *types.Builtin:
			return obj.Name()
		default:
			return ""
		}
	default:
		return ""
	}
}

func IsCallToAST(pass *analysis.Pass, node ast.Node, name string) bool {
	call, ok := node.(*ast.CallExpr)
	if !ok {
		return false
	}
	return CallNameAST(pass, call) == name
}

func IsCallToAnyAST(pass *analysis.Pass, node ast.Node, names ...string) bool {
	call, ok := node.(*ast.CallExpr)
	if !ok {
		return false
	}
	q := CallNameAST(pass, call)
	for _, name := range names {
		if q == name {
			return true
		}
	}
	return false
}

func Preamble(f *ast.File) string {
	cutoff := f.Package
	if f.Doc != nil {
		cutoff = f.Doc.Pos()
	}
	var out []string
	for _, cmt := range f.Comments {
		if cmt.Pos() >= cutoff {
			break
		}
		out = append(out, cmt.Text())
	}
	return strings.Join(out, "\n")
}

func GroupSpecs(fset *token.FileSet, specs []ast.Spec) [][]ast.Spec {
	if len(specs) == 0 {
		return nil
	}
	groups := make([][]ast.Spec, 1)
	groups[0] = append(groups[0], specs[0])

	for _, spec := range specs[1:] {
		g := groups[len(groups)-1]
		if fset.PositionFor(spec.Pos(), false).Line-1 !=
			fset.PositionFor(g[len(g)-1].End(), false).Line {

			groups = append(groups, nil)
		}

		groups[len(groups)-1] = append(groups[len(groups)-1], spec)
	}

	return groups
}

func IsObject(obj types.Object, name string) bool {
	var path string
	if pkg := obj.Pkg(); pkg != nil {
		path = pkg.Path() + "."
	}
	return path+obj.Name() == name
}

type Field struct {
	Var  *types.Var
	Tag  string
	Path []int
}

// FlattenFields recursively flattens T and embedded structs,
// returning a list of fields. If multiple fields with the same name
// exist, all will be returned.
func FlattenFields(T *types.Struct) []Field {
	return flattenFields(T, nil, nil)
}

func flattenFields(T *types.Struct, path []int, seen map[types.Type]bool) []Field {
	if seen == nil {
		seen = map[types.Type]bool{}
	}
	if seen[T] {
		return nil
	}
	seen[T] = true
	var out []Field
	for i := 0; i < T.NumFields(); i++ {
		field := T.Field(i)
		tag := T.Tag(i)
		np := append(path[:len(path):len(path)], i)
		if field.Anonymous() {
			if s, ok := Dereference(field.Type()).Underlying().(*types.Struct); ok {
				out = append(out, flattenFields(s, np, seen)...)
			}
		} else {
			out = append(out, Field{field, tag, np})
		}
	}
	return out
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
		// the literal itself cannot have side ffects, only calling it
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
	version := pass.Analyzer.Flags.Lookup("go").Value.(flag.Getter).Get().(int)
	return version >= minor
}

func Preorder(pass *analysis.Pass, fn func(ast.Node), types ...ast.Node) {
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder(types, fn)
}
