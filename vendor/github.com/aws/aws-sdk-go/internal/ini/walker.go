package ini

// Walk will traverse the AST using the v, the Visitor.
func Walk(tree []AST, v Visitor) error {
	for _, node := range tree {
		switch node.Kind {
		case ASTKindExpr,
			ASTKindExprStatement:

			if err := v.VisitExpr(node); err != nil {
				return err
			}
		case ASTKindStatement,
			ASTKindCompletedSectionStatement,
			ASTKindNestedSectionStatement,
			ASTKindCompletedNestedSectionStatement:

			if err := v.VisitStatement(node); err != nil {
				return err
			}
		}
	}

	return nil
}
