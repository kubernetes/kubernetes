// Package lintdsl provides helpers for implementing static analysis
// checks. Dot-importing this package is encouraged.
package lintdsl

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/constant"
	"go/printer"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"honnef.co/go/tools/facts"
	"honnef.co/go/tools/lint"
	"honnef.co/go/tools/ssa"
)

type packager interface {
	Package() *ssa.Package
}

func CallName(call *ssa.CallCommon) string {
	if call.IsInvoke() {
		return ""
	}
	switch v := call.Value.(type) {
	case *ssa.Function:
		fn, ok := v.Object().(*types.Func)
		if !ok {
			return ""
		}
		return lint.FuncName(fn)
	case *ssa.Builtin:
		return v.Name()
	}
	return ""
}

func IsCallTo(call *ssa.CallCommon, name string) bool { return CallName(call) == name }
func IsType(T types.Type, name string) bool           { return types.TypeString(T, nil) == name }

func FilterDebug(instr []ssa.Instruction) []ssa.Instruction {
	var out []ssa.Instruction
	for _, ins := range instr {
		if _, ok := ins.(*ssa.DebugRef); !ok {
			out = append(out, ins)
		}
	}
	return out
}

func IsExample(fn *ssa.Function) bool {
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

func IsInTest(pass *analysis.Pass, node lint.Positioner) bool {
	// FIXME(dh): this doesn't work for global variables with
	// initializers
	f := pass.Fset.File(node.Pos())
	return f != nil && strings.HasSuffix(f.Name(), "_test.go")
}

func IsInMain(pass *analysis.Pass, node lint.Positioner) bool {
	if node, ok := node.(packager); ok {
		return node.Package().Pkg.Name() == "main"
	}
	return pass.Pkg.Name() == "main"
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

func IsGoVersion(pass *analysis.Pass, minor int) bool {
	version := pass.Analyzer.Flags.Lookup("go").Value.(flag.Getter).Get().(int)
	return version >= minor
}

func CallNameAST(pass *analysis.Pass, call *ast.CallExpr) string {
	switch fun := call.Fun.(type) {
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
	for _, name := range names {
		if IsCallToAST(pass, node, name) {
			return true
		}
	}
	return false
}

func Render(pass *analysis.Pass, x interface{}) string {
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, pass.Fset, x); err != nil {
		panic(err)
	}
	return buf.String()
}

func RenderArgs(pass *analysis.Pass, args []ast.Expr) string {
	var ss []string
	for _, arg := range args {
		ss = append(ss, Render(pass, arg))
	}
	return strings.Join(ss, ", ")
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

func Inspect(node ast.Node, fn func(node ast.Node) bool) {
	if node == nil {
		return
	}
	ast.Inspect(node, fn)
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

func File(pass *analysis.Pass, node lint.Positioner) *ast.File {
	pass.Fset.PositionFor(node.Pos(), true)
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

func ReportfFG(pass *analysis.Pass, pos token.Pos, f string, args ...interface{}) {
	file := lint.DisplayPosition(pass.Fset, pos).Filename
	m := pass.ResultOf[facts.Generated].(map[string]facts.Generator)
	if _, ok := m[file]; ok {
		return
	}
	pass.Reportf(pos, f, args...)
}

func ReportNodef(pass *analysis.Pass, node ast.Node, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	pass.Report(analysis.Diagnostic{Pos: node.Pos(), End: node.End(), Message: msg})
}

func ReportNodefFG(pass *analysis.Pass, node ast.Node, format string, args ...interface{}) {
	file := lint.DisplayPosition(pass.Fset, node.Pos()).Filename
	m := pass.ResultOf[facts.Generated].(map[string]facts.Generator)
	if _, ok := m[file]; ok {
		return
	}
	ReportNodef(pass, node, format, args...)
}
