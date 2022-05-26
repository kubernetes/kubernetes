package astutil

import (
	"fmt"
	"go/ast"
	"go/token"
	"reflect"
	"strings"
)

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

// Unparen returns e with any enclosing parentheses stripped.
func Unparen(e ast.Expr) ast.Expr {
	for {
		p, ok := e.(*ast.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}

func CopyExpr(node ast.Expr) ast.Expr {
	switch node := node.(type) {
	case *ast.BasicLit:
		cp := *node
		return &cp
	case *ast.BinaryExpr:
		cp := *node
		cp.X = CopyExpr(cp.X)
		cp.Y = CopyExpr(cp.Y)
		return &cp
	case *ast.CallExpr:
		cp := *node
		cp.Fun = CopyExpr(cp.Fun)
		cp.Args = make([]ast.Expr, len(node.Args))
		for i, v := range node.Args {
			cp.Args[i] = CopyExpr(v)
		}
		return &cp
	case *ast.CompositeLit:
		cp := *node
		cp.Type = CopyExpr(cp.Type)
		cp.Elts = make([]ast.Expr, len(node.Elts))
		for i, v := range node.Elts {
			cp.Elts[i] = CopyExpr(v)
		}
		return &cp
	case *ast.Ident:
		cp := *node
		return &cp
	case *ast.IndexExpr:
		cp := *node
		cp.X = CopyExpr(cp.X)
		cp.Index = CopyExpr(cp.Index)
		return &cp
	case *ast.KeyValueExpr:
		cp := *node
		cp.Key = CopyExpr(cp.Key)
		cp.Value = CopyExpr(cp.Value)
		return &cp
	case *ast.ParenExpr:
		cp := *node
		cp.X = CopyExpr(cp.X)
		return &cp
	case *ast.SelectorExpr:
		cp := *node
		cp.X = CopyExpr(cp.X)
		cp.Sel = CopyExpr(cp.Sel).(*ast.Ident)
		return &cp
	case *ast.SliceExpr:
		cp := *node
		cp.X = CopyExpr(cp.X)
		cp.Low = CopyExpr(cp.Low)
		cp.High = CopyExpr(cp.High)
		cp.Max = CopyExpr(cp.Max)
		return &cp
	case *ast.StarExpr:
		cp := *node
		cp.X = CopyExpr(cp.X)
		return &cp
	case *ast.TypeAssertExpr:
		cp := *node
		cp.X = CopyExpr(cp.X)
		cp.Type = CopyExpr(cp.Type)
		return &cp
	case *ast.UnaryExpr:
		cp := *node
		cp.X = CopyExpr(cp.X)
		return &cp
	case *ast.MapType:
		cp := *node
		cp.Key = CopyExpr(cp.Key)
		cp.Value = CopyExpr(cp.Value)
		return &cp
	case *ast.ArrayType:
		cp := *node
		cp.Len = CopyExpr(cp.Len)
		cp.Elt = CopyExpr(cp.Elt)
		return &cp
	case *ast.Ellipsis:
		cp := *node
		cp.Elt = CopyExpr(cp.Elt)
		return &cp
	case *ast.InterfaceType:
		cp := *node
		return &cp
	case *ast.StructType:
		cp := *node
		return &cp
	case *ast.FuncLit:
		// TODO(dh): implement copying of function literals.
		return nil
	case *ast.ChanType:
		cp := *node
		cp.Value = CopyExpr(cp.Value)
		return &cp
	case nil:
		return nil
	default:
		panic(fmt.Sprintf("unreachable: %T", node))
	}
}

func Equal(a, b ast.Node) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if reflect.TypeOf(a) != reflect.TypeOf(b) {
		return false
	}

	switch a := a.(type) {
	case *ast.BasicLit:
		b := b.(*ast.BasicLit)
		return a.Kind == b.Kind && a.Value == b.Value
	case *ast.BinaryExpr:
		b := b.(*ast.BinaryExpr)
		return Equal(a.X, b.X) && a.Op == b.Op && Equal(a.Y, b.Y)
	case *ast.CallExpr:
		b := b.(*ast.CallExpr)
		if len(a.Args) != len(b.Args) {
			return false
		}
		for i, arg := range a.Args {
			if !Equal(arg, b.Args[i]) {
				return false
			}
		}
		return Equal(a.Fun, b.Fun) &&
			(a.Ellipsis == token.NoPos && b.Ellipsis == token.NoPos || a.Ellipsis != token.NoPos && b.Ellipsis != token.NoPos)
	case *ast.CompositeLit:
		b := b.(*ast.CompositeLit)
		if len(a.Elts) != len(b.Elts) {
			return false
		}
		for i, elt := range b.Elts {
			if !Equal(elt, b.Elts[i]) {
				return false
			}
		}
		return Equal(a.Type, b.Type) && a.Incomplete == b.Incomplete
	case *ast.Ident:
		b := b.(*ast.Ident)
		return a.Name == b.Name
	case *ast.IndexExpr:
		b := b.(*ast.IndexExpr)
		return Equal(a.X, b.X) && Equal(a.Index, b.Index)
	case *ast.KeyValueExpr:
		b := b.(*ast.KeyValueExpr)
		return Equal(a.Key, b.Key) && Equal(a.Value, b.Value)
	case *ast.ParenExpr:
		b := b.(*ast.ParenExpr)
		return Equal(a.X, b.X)
	case *ast.SelectorExpr:
		b := b.(*ast.SelectorExpr)
		return Equal(a.X, b.X) && Equal(a.Sel, b.Sel)
	case *ast.SliceExpr:
		b := b.(*ast.SliceExpr)
		return Equal(a.X, b.X) && Equal(a.Low, b.Low) && Equal(a.High, b.High) && Equal(a.Max, b.Max) && a.Slice3 == b.Slice3
	case *ast.StarExpr:
		b := b.(*ast.StarExpr)
		return Equal(a.X, b.X)
	case *ast.TypeAssertExpr:
		b := b.(*ast.TypeAssertExpr)
		return Equal(a.X, b.X) && Equal(a.Type, b.Type)
	case *ast.UnaryExpr:
		b := b.(*ast.UnaryExpr)
		return a.Op == b.Op && Equal(a.X, b.X)
	case *ast.MapType:
		b := b.(*ast.MapType)
		return Equal(a.Key, b.Key) && Equal(a.Value, b.Value)
	case *ast.ArrayType:
		b := b.(*ast.ArrayType)
		return Equal(a.Len, b.Len) && Equal(a.Elt, b.Elt)
	case *ast.Ellipsis:
		b := b.(*ast.Ellipsis)
		return Equal(a.Elt, b.Elt)
	case *ast.InterfaceType:
		b := b.(*ast.InterfaceType)
		return a.Incomplete == b.Incomplete && Equal(a.Methods, b.Methods)
	case *ast.StructType:
		b := b.(*ast.StructType)
		return a.Incomplete == b.Incomplete && Equal(a.Fields, b.Fields)
	case *ast.FuncLit:
		// TODO(dh): support function literals
		return false
	case *ast.ChanType:
		b := b.(*ast.ChanType)
		return a.Dir == b.Dir && (a.Arrow == token.NoPos && b.Arrow == token.NoPos || a.Arrow != token.NoPos && b.Arrow != token.NoPos)
	case *ast.FieldList:
		b := b.(*ast.FieldList)
		if len(a.List) != len(b.List) {
			return false
		}
		for i, fieldA := range a.List {
			if !Equal(fieldA, b.List[i]) {
				return false
			}
		}
		return true
	case *ast.Field:
		b := b.(*ast.Field)
		if len(a.Names) != len(b.Names) {
			return false
		}
		for j, name := range a.Names {
			if !Equal(name, b.Names[j]) {
				return false
			}
		}
		if !Equal(a.Type, b.Type) || !Equal(a.Tag, b.Tag) {
			return false
		}
		return true
	default:
		panic(fmt.Sprintf("unreachable: %T", a))
	}
}
