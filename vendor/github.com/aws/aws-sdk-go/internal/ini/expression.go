package ini

// newExpression will return an expression AST.
// Expr represents an expression
//
//	grammar:
//	expr -> string | number
func newExpression(tok Token) AST {
	return newASTWithRootToken(ASTKindExpr, tok)
}

func newEqualExpr(left AST, tok Token) AST {
	return newASTWithRootToken(ASTKindEqualExpr, tok, left)
}

// EqualExprKey will return a LHS value in the equal expr
func EqualExprKey(ast AST) string {
	children := ast.GetChildren()
	if len(children) == 0 || ast.Kind != ASTKindEqualExpr {
		return ""
	}

	return string(children[0].Root.Raw())
}
