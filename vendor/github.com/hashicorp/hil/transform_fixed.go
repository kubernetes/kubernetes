package hil

import (
	"github.com/hashicorp/hil/ast"
)

// FixedValueTransform transforms an AST to return a fixed value for
// all interpolations. i.e. you can make "hi ${anything}" always
// turn into "hi foo".
//
// The primary use case for this is for config validations where you can
// verify that interpolations result in a certain type of string.
func FixedValueTransform(root ast.Node, Value *ast.LiteralNode) ast.Node {
	// We visit the nodes in top-down order
	result := root
	switch n := result.(type) {
	case *ast.Output:
		for i, v := range n.Exprs {
			n.Exprs[i] = FixedValueTransform(v, Value)
		}
	case *ast.LiteralNode:
		// We keep it as-is
	default:
		// Anything else we replace
		result = Value
	}

	return result
}
