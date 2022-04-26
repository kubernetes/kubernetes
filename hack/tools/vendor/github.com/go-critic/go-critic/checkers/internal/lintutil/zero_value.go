package lintutil

import (
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
)

// IsZeroValue reports whether x represents zero value of its type.
//
// The functions is conservative and may return false for zero values
// if some cases are not handled in a comprehensive way
// but is should never return true for something that's not a proper zv.
func IsZeroValue(info *types.Info, x ast.Expr) bool {
	switch x := x.(type) {
	case *ast.BasicLit:
		typ := info.TypeOf(x).Underlying().(*types.Basic)
		v := info.Types[x].Value
		var z constant.Value
		switch {
		case typ.Kind() == types.String:
			z = constant.MakeString("")
		case typ.Info()&types.IsInteger != 0:
			z = constant.MakeInt64(0)
		case typ.Info()&types.IsUnsigned != 0:
			z = constant.MakeUint64(0)
		case typ.Info()&types.IsFloat != 0:
			z = constant.MakeFloat64(0)
		default:
			return false
		}
		return constant.Compare(v, token.EQL, z)

	case *ast.CompositeLit:
		return len(x.Elts) == 0

	default:
		// Note that this function is not comprehensive.
		return false
	}
}

// ZeroValueOf returns a zero value expression for typeExpr of type typ.
// If function can't find such a value, nil is returned.
func ZeroValueOf(typeExpr ast.Expr, typ types.Type) ast.Expr {
	switch utyp := typ.Underlying().(type) {
	case *types.Basic:
		info := utyp.Info()
		var zv ast.Expr
		switch {
		case info&types.IsInteger != 0:
			zv = &ast.BasicLit{Kind: token.INT, Value: "0"}
		case info&types.IsFloat != 0:
			zv = &ast.BasicLit{Kind: token.FLOAT, Value: "0.0"}
		case info&types.IsString != 0:
			zv = &ast.BasicLit{Kind: token.STRING, Value: `""`}
		case info&types.IsBoolean != 0:
			zv = &ast.Ident{Name: "false"}
		}
		if isDefaultLiteralType(typ) {
			return zv
		}
		return &ast.CallExpr{
			Fun:  typeExpr,
			Args: []ast.Expr{zv},
		}

	case *types.Slice, *types.Map, *types.Pointer, *types.Interface:
		return &ast.CallExpr{
			Fun:  typeExpr,
			Args: []ast.Expr{&ast.Ident{Name: "nil"}},
		}

	case *types.Array, *types.Struct:
		return &ast.CompositeLit{Type: typeExpr}

	default:
		return nil
	}
}

func isDefaultLiteralType(typ types.Type) bool {
	btyp, ok := typ.(*types.Basic)
	if !ok {
		return false
	}
	switch btyp.Kind() {
	case types.Bool, types.Int, types.Float64, types.String:
		return true
	default:
		return false
	}
}
