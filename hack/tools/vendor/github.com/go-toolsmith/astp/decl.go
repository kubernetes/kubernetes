package astp

import "go/ast"

// IsDecl reports whether a node is a ast.Decl.
func IsDecl(node ast.Node) bool {
	_, ok := node.(ast.Decl)
	return ok
}

// IsFuncDecl reports whether a given ast.Node is a function declaration (*ast.FuncDecl).
func IsFuncDecl(node ast.Node) bool {
	_, ok := node.(*ast.FuncDecl)
	return ok
}

// IsGenDecl reports whether a given ast.Node is a generic declaration (*ast.GenDecl).
func IsGenDecl(node ast.Node) bool {
	_, ok := node.(*ast.GenDecl)
	return ok
}

// IsImportSpec reports whether a given ast.Node is an import declaration (*ast.ImportSpec).
func IsImportSpec(node ast.Node) bool {
	_, ok := node.(*ast.ImportSpec)
	return ok
}

// IsValueSpec reports whether a given ast.Node is a value declaration (*ast.ValueSpec).
func IsValueSpec(node ast.Node) bool {
	_, ok := node.(*ast.ValueSpec)
	return ok
}

// IsTypeSpec reports whether a given ast.Node is a type declaration (*ast.TypeSpec).
func IsTypeSpec(node ast.Node) bool {
	_, ok := node.(*ast.TypeSpec)
	return ok
}
