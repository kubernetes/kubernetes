package ini

// Statement is an empty AST mostly used for transitioning states.
func newStatement() AST {
	return newAST(ASTKindStatement, AST{})
}

// SectionStatement represents a section AST
func newSectionStatement(tok Token) AST {
	return newASTWithRootToken(ASTKindSectionStatement, tok)
}

// ExprStatement represents a completed expression AST
func newExprStatement(ast AST) AST {
	return newAST(ASTKindExprStatement, ast)
}

// CommentStatement represents a comment in the ini definition.
//
//	grammar:
//	comment -> #comment' | ;comment'
//	comment' -> epsilon | value
func newCommentStatement(tok Token) AST {
	return newAST(ASTKindCommentStatement, newExpression(tok))
}

// CompletedSectionStatement represents a completed section
func newCompletedSectionStatement(ast AST) AST {
	return newAST(ASTKindCompletedSectionStatement, ast)
}

// SkipStatement is used to skip whole statements
func newSkipStatement(ast AST) AST {
	return newAST(ASTKindSkipStatement, ast)
}
