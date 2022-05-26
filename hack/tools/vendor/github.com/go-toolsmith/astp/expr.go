package astp

import "go/ast"

// IsExpr reports whether a given ast.Node is an expression(ast.Expr).
func IsExpr(node ast.Node) bool {
	_, ok := node.(ast.Expr)
	return ok
}

// IsBadExpr reports whether a given ast.Node is a bad expression (*ast.IsBadExpr).
func IsBadExpr(node ast.Node) bool {
	_, ok := node.(*ast.BadExpr)
	return ok
}

// IsIdent reports whether a given ast.Node is an identifier (*ast.IsIdent).
func IsIdent(node ast.Node) bool {
	_, ok := node.(*ast.Ident)
	return ok
}

// IsEllipsis reports whether a given ast.Node is an `...` (ellipsis) (*ast.IsEllipsis).
func IsEllipsis(node ast.Node) bool {
	_, ok := node.(*ast.Ellipsis)
	return ok
}

// IsBasicLit reports whether a given ast.Node is a literal of basic type (*ast.IsBasicLit).
func IsBasicLit(node ast.Node) bool {
	_, ok := node.(*ast.BasicLit)
	return ok
}

// IsFuncLit reports whether a given ast.Node is a function literal (*ast.IsFuncLit).
func IsFuncLit(node ast.Node) bool {
	_, ok := node.(*ast.FuncLit)
	return ok
}

// IsCompositeLit reports whether a given ast.Node is a composite literal (*ast.IsCompositeLit).
func IsCompositeLit(node ast.Node) bool {
	_, ok := node.(*ast.CompositeLit)
	return ok
}

// IsParenExpr reports whether a given ast.Node is a parenthesized expression (*ast.IsParenExpr).
func IsParenExpr(node ast.Node) bool {
	_, ok := node.(*ast.ParenExpr)
	return ok
}

// IsSelectorExpr reports whether a given ast.Node is a selector expression (*ast.IsSelectorExpr).
func IsSelectorExpr(node ast.Node) bool {
	_, ok := node.(*ast.SelectorExpr)
	return ok
}

// IsIndexExpr reports whether a given ast.Node is an index expression (*ast.IsIndexExpr).
func IsIndexExpr(node ast.Node) bool {
	_, ok := node.(*ast.IndexExpr)
	return ok
}

// IsSliceExpr reports whether a given ast.Node is a slice expression (*ast.IsSliceExpr).
func IsSliceExpr(node ast.Node) bool {
	_, ok := node.(*ast.SliceExpr)
	return ok
}

// IsTypeAssertExpr reports whether a given ast.Node is a type assert expression (*ast.IsTypeAssertExpr).
func IsTypeAssertExpr(node ast.Node) bool {
	_, ok := node.(*ast.TypeAssertExpr)
	return ok
}

// IsCallExpr reports whether a given ast.Node is an expression followed by an argument list (*ast.IsCallExpr).
func IsCallExpr(node ast.Node) bool {
	_, ok := node.(*ast.CallExpr)
	return ok
}

// IsStarExpr reports whether a given ast.Node is a star expression(unary "*" or apointer) (*ast.IsStarExpr)
func IsStarExpr(node ast.Node) bool {
	_, ok := node.(*ast.StarExpr)
	return ok
}

// IsUnaryExpr reports whether a given ast.Node is a unary expression (*ast.IsUnaryExpr).
func IsUnaryExpr(node ast.Node) bool {
	_, ok := node.(*ast.UnaryExpr)
	return ok
}

// IsBinaryExpr reports whether a given ast.Node is a binary expression (*ast.IsBinaryExpr).
func IsBinaryExpr(node ast.Node) bool {
	_, ok := node.(*ast.BinaryExpr)
	return ok
}

// IsKeyValueExpr reports whether a given ast.Node is a (key:value) pair (*ast.IsKeyValueExpr).
func IsKeyValueExpr(node ast.Node) bool {
	_, ok := node.(*ast.KeyValueExpr)
	return ok
}

// IsArrayType reports whether a given ast.Node is an array or slice type (*ast.IsArrayType).
func IsArrayType(node ast.Node) bool {
	_, ok := node.(*ast.ArrayType)
	return ok
}

// IsStructType reports whether a given ast.Node is a struct type (*ast.IsStructType).
func IsStructType(node ast.Node) bool {
	_, ok := node.(*ast.StructType)
	return ok
}

// IsFuncType reports whether a given ast.Node is a function type (*ast.IsFuncType).
func IsFuncType(node ast.Node) bool {
	_, ok := node.(*ast.FuncType)
	return ok
}

// IsInterfaceType reports whether a given ast.Node is an interface type (*ast.IsInterfaceType).
func IsInterfaceType(node ast.Node) bool {
	_, ok := node.(*ast.InterfaceType)
	return ok
}

// IsMapType reports whether a given ast.Node is a map type (*ast.IsMapType).
func IsMapType(node ast.Node) bool {
	_, ok := node.(*ast.MapType)
	return ok
}

// IsChanType reports whether a given ast.Node is a channel type (*ast.IsChanType).
func IsChanType(node ast.Node) bool {
	_, ok := node.(*ast.ChanType)
	return ok
}
