package astp

import "go/ast"

// IsStmt reports whether a given ast.Node is a statement(ast.Stmt).
func IsStmt(node ast.Node) bool {
	_, ok := node.(ast.Stmt)
	return ok
}

// IsBadStmt reports whether a given ast.Node is a bad statement(*ast.BadStmt)
func IsBadStmt(node ast.Node) bool {
	_, ok := node.(*ast.BadStmt)
	return ok
}

// IsDeclStmt reports whether a given ast.Node is a declaration statement(*ast.DeclStmt)
func IsDeclStmt(node ast.Node) bool {
	_, ok := node.(*ast.DeclStmt)
	return ok
}

// IsEmptyStmt reports whether a given ast.Node is an empty statement(*ast.EmptyStmt)
func IsEmptyStmt(node ast.Node) bool {
	_, ok := node.(*ast.EmptyStmt)
	return ok
}

// IsLabeledStmt reports whether a given ast.Node is a label statement(*ast.LabeledStmt)
func IsLabeledStmt(node ast.Node) bool {
	_, ok := node.(*ast.LabeledStmt)
	return ok
}

// IsExprStmt reports whether a given ast.Node is an expression statement(*ast.ExprStmt)
func IsExprStmt(node ast.Node) bool {
	_, ok := node.(*ast.ExprStmt)
	return ok
}

// IsSendStmt reports whether a given ast.Node is a send to chan statement(*ast.SendStmt)
func IsSendStmt(node ast.Node) bool {
	_, ok := node.(*ast.SendStmt)
	return ok
}

// IsIncDecStmt reports whether a given ast.Node is a increment/decrement statement(*ast.IncDecStmt)
func IsIncDecStmt(node ast.Node) bool {
	_, ok := node.(*ast.IncDecStmt)
	return ok
}

// IsAssignStmt reports whether a given ast.Node is an assignment statement(*ast.AssignStmt)
func IsAssignStmt(node ast.Node) bool {
	_, ok := node.(*ast.AssignStmt)
	return ok
}

// IsGoStmt reports whether a given ast.Node is a go statement(*ast.GoStmt)
func IsGoStmt(node ast.Node) bool {
	_, ok := node.(*ast.GoStmt)
	return ok
}

// IsDeferStmt reports whether a given ast.Node is a defer statement(*ast.DeferStmt)
func IsDeferStmt(node ast.Node) bool {
	_, ok := node.(*ast.DeferStmt)
	return ok
}

// IsReturnStmt reports whether a given ast.Node is a return statement(*ast.ReturnStmt)
func IsReturnStmt(node ast.Node) bool {
	_, ok := node.(*ast.ReturnStmt)
	return ok
}

// IsBranchStmt reports whether a given ast.Node is a branch(goto/continue/break/fallthrough)statement(*ast.BranchStmt)
func IsBranchStmt(node ast.Node) bool {
	_, ok := node.(*ast.BranchStmt)
	return ok
}

// IsBlockStmt reports whether a given ast.Node is a block statement(*ast.BlockStmt)
func IsBlockStmt(node ast.Node) bool {
	_, ok := node.(*ast.BlockStmt)
	return ok
}

// IsIfStmt reports whether a given ast.Node is an if statement(*ast.IfStmt)
func IsIfStmt(node ast.Node) bool {
	_, ok := node.(*ast.IfStmt)
	return ok
}

// IsCaseClause reports whether a given ast.Node is a case statement(*ast.CaseClause)
func IsCaseClause(node ast.Node) bool {
	_, ok := node.(*ast.CaseClause)
	return ok
}

// IsSwitchStmt reports whether a given ast.Node is a switch statement(*ast.SwitchStmt)
func IsSwitchStmt(node ast.Node) bool {
	_, ok := node.(*ast.SwitchStmt)
	return ok
}

// IsTypeSwitchStmt reports whether a given ast.Node is a type switch statement(*ast.TypeSwitchStmt)
func IsTypeSwitchStmt(node ast.Node) bool {
	_, ok := node.(*ast.TypeSwitchStmt)
	return ok
}

// IsCommClause reports whether a given ast.Node is a select statement(*ast.CommClause)
func IsCommClause(node ast.Node) bool {
	_, ok := node.(*ast.CommClause)
	return ok
}

// IsSelectStmt reports whether a given ast.Node is a selection statement(*ast.SelectStmt)
func IsSelectStmt(node ast.Node) bool {
	_, ok := node.(*ast.SelectStmt)
	return ok
}

// IsForStmt reports whether a given ast.Node is a for statement(*ast.ForStmt)
func IsForStmt(node ast.Node) bool {
	_, ok := node.(*ast.ForStmt)
	return ok
}

// IsRangeStmt reports whether a given ast.Node is a range statement(*ast.RangeStmt)
func IsRangeStmt(node ast.Node) bool {
	_, ok := node.(*ast.RangeStmt)
	return ok
}
