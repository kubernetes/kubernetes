package typep

import (
	"go/ast"
	"go/types"
)

// IsTypeExpr reports whether x represents a type expression.
//
// Type expression does not evaluate to any run time value,
// but rather describes a type that is used inside Go expression.
//
// For example, (*T)(v) is a CallExpr that "calls" (*T).
// (*T) is a type expression that tells Go compiler type v should be converted to.
func IsTypeExpr(info *types.Info, x ast.Expr) bool {
	switch x := x.(type) {
	case *ast.StarExpr:
		return IsTypeExpr(info, x.X)
	case *ast.ParenExpr:
		return IsTypeExpr(info, x.X)
	case *ast.SelectorExpr:
		return IsTypeExpr(info, x.Sel)

	case *ast.Ident:
		// Identifier may be a type expression if object
		// it reffers to is a type name.
		_, ok := info.ObjectOf(x).(*types.TypeName)
		return ok

	case *ast.FuncType, *ast.StructType, *ast.InterfaceType, *ast.ArrayType, *ast.MapType, *ast.ChanType:
		return true

	default:
		return false
	}
}
