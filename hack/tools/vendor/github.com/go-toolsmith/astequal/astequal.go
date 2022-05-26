// Package astequal provides AST (deep) equallity check operations.
package astequal

import (
	"go/ast"
	"go/token"
)

// Node reports whether two AST nodes are structurally (deep) equal.
//
// Nil arguments are permitted: true is returned if x and y are both nils.
//
// See also: Expr, Stmt, Decl functions.
func Node(x, y ast.Node) bool {
	return astNodeEq(x, y)
}

// Expr reports whether two AST expressions are structurally (deep) equal.
//
// Nil arguments are permitted: true is returned if x and y are both nils.
// ast.BadExpr comparison always yields false.
func Expr(x, y ast.Expr) bool {
	return astExprEq(x, y)
}

// Stmt reports whether two AST statements are structurally (deep) equal.
//
// Nil arguments are permitted: true is returned if x and y are both nils.
// ast.BadStmt comparison always yields false.
func Stmt(x, y ast.Stmt) bool {
	return astStmtEq(x, y)
}

// Decl reports whether two AST declarations are structurally (deep) equal.
//
// Nil arguments are permitted: true is returned if x and y are both nils.
// ast.BadDecl comparison always yields false.
func Decl(x, y ast.Decl) bool {
	return astDeclEq(x, y)
}

// Functions to perform deep equallity checks between arbitrary AST nodes.

// Compare interface node types.
//
// Interfaces, as well as their values, can be nil.
//
// Even if AST does expect field X to be mandatory,
// nil checks are required as nodes can be constructed
// manually, or be partially invalid/incomplete.

func astNodeEq(x, y ast.Node) bool {
	switch x := x.(type) {
	case ast.Expr:
		y, ok := y.(ast.Expr)
		return ok && astExprEq(x, y)
	case ast.Stmt:
		y, ok := y.(ast.Stmt)
		return ok && astStmtEq(x, y)
	case ast.Decl:
		y, ok := y.(ast.Decl)
		return ok && astDeclEq(x, y)

	case *ast.Field:
		y, ok := y.(*ast.Field)
		return ok && astFieldEq(x, y)
	case *ast.FieldList:
		y, ok := y.(*ast.FieldList)
		return ok && astFieldListEq(x, y)

	default:
		return false
	}
}

func astExprEq(x, y ast.Expr) bool {
	if x == nil || y == nil {
		return x == y
	}

	switch x := x.(type) {
	case *ast.Ident:
		y, ok := y.(*ast.Ident)
		return ok && astIdentEq(x, y)

	case *ast.BasicLit:
		y, ok := y.(*ast.BasicLit)
		return ok && astBasicLitEq(x, y)

	case *ast.FuncLit:
		y, ok := y.(*ast.FuncLit)
		return ok && astFuncLitEq(x, y)

	case *ast.CompositeLit:
		y, ok := y.(*ast.CompositeLit)
		return ok && astCompositeLitEq(x, y)

	case *ast.ParenExpr:
		y, ok := y.(*ast.ParenExpr)
		return ok && astParenExprEq(x, y)

	case *ast.SelectorExpr:
		y, ok := y.(*ast.SelectorExpr)
		return ok && astSelectorExprEq(x, y)

	case *ast.IndexExpr:
		y, ok := y.(*ast.IndexExpr)
		return ok && astIndexExprEq(x, y)

	case *ast.SliceExpr:
		y, ok := y.(*ast.SliceExpr)
		return ok && astSliceExprEq(x, y)

	case *ast.TypeAssertExpr:
		y, ok := y.(*ast.TypeAssertExpr)
		return ok && astTypeAssertExprEq(x, y)

	case *ast.CallExpr:
		y, ok := y.(*ast.CallExpr)
		return ok && astCallExprEq(x, y)

	case *ast.StarExpr:
		y, ok := y.(*ast.StarExpr)
		return ok && astStarExprEq(x, y)

	case *ast.UnaryExpr:
		y, ok := y.(*ast.UnaryExpr)
		return ok && astUnaryExprEq(x, y)

	case *ast.BinaryExpr:
		y, ok := y.(*ast.BinaryExpr)
		return ok && astBinaryExprEq(x, y)

	case *ast.KeyValueExpr:
		y, ok := y.(*ast.KeyValueExpr)
		return ok && astKeyValueExprEq(x, y)

	case *ast.ArrayType:
		y, ok := y.(*ast.ArrayType)
		return ok && astArrayTypeEq(x, y)

	case *ast.StructType:
		y, ok := y.(*ast.StructType)
		return ok && astStructTypeEq(x, y)

	case *ast.FuncType:
		y, ok := y.(*ast.FuncType)
		return ok && astFuncTypeEq(x, y)

	case *ast.InterfaceType:
		y, ok := y.(*ast.InterfaceType)
		return ok && astInterfaceTypeEq(x, y)

	case *ast.MapType:
		y, ok := y.(*ast.MapType)
		return ok && astMapTypeEq(x, y)

	case *ast.ChanType:
		y, ok := y.(*ast.ChanType)
		return ok && astChanTypeEq(x, y)

	case *ast.Ellipsis:
		y, ok := y.(*ast.Ellipsis)
		return ok && astEllipsisEq(x, y)

	default:
		return false
	}
}

func astStmtEq(x, y ast.Stmt) bool {
	if x == nil || y == nil {
		return x == y
	}

	switch x := x.(type) {
	case *ast.ExprStmt:
		y, ok := y.(*ast.ExprStmt)
		return ok && astExprStmtEq(x, y)

	case *ast.SendStmt:
		y, ok := y.(*ast.SendStmt)
		return ok && astSendStmtEq(x, y)

	case *ast.IncDecStmt:
		y, ok := y.(*ast.IncDecStmt)
		return ok && astIncDecStmtEq(x, y)

	case *ast.AssignStmt:
		y, ok := y.(*ast.AssignStmt)
		return ok && astAssignStmtEq(x, y)

	case *ast.GoStmt:
		y, ok := y.(*ast.GoStmt)
		return ok && astGoStmtEq(x, y)

	case *ast.DeferStmt:
		y, ok := y.(*ast.DeferStmt)
		return ok && astDeferStmtEq(x, y)

	case *ast.ReturnStmt:
		y, ok := y.(*ast.ReturnStmt)
		return ok && astReturnStmtEq(x, y)

	case *ast.BranchStmt:
		y, ok := y.(*ast.BranchStmt)
		return ok && astBranchStmtEq(x, y)

	case *ast.BlockStmt:
		y, ok := y.(*ast.BlockStmt)
		return ok && astBlockStmtEq(x, y)

	case *ast.IfStmt:
		y, ok := y.(*ast.IfStmt)
		return ok && astIfStmtEq(x, y)

	case *ast.CaseClause:
		y, ok := y.(*ast.CaseClause)
		return ok && astCaseClauseEq(x, y)

	case *ast.SwitchStmt:
		y, ok := y.(*ast.SwitchStmt)
		return ok && astSwitchStmtEq(x, y)

	case *ast.TypeSwitchStmt:
		y, ok := y.(*ast.TypeSwitchStmt)
		return ok && astTypeSwitchStmtEq(x, y)

	case *ast.CommClause:
		y, ok := y.(*ast.CommClause)
		return ok && astCommClauseEq(x, y)

	case *ast.SelectStmt:
		y, ok := y.(*ast.SelectStmt)
		return ok && astSelectStmtEq(x, y)

	case *ast.ForStmt:
		y, ok := y.(*ast.ForStmt)
		return ok && astForStmtEq(x, y)

	case *ast.RangeStmt:
		y, ok := y.(*ast.RangeStmt)
		return ok && astRangeStmtEq(x, y)

	case *ast.DeclStmt:
		y, ok := y.(*ast.DeclStmt)
		return ok && astDeclStmtEq(x, y)

	case *ast.LabeledStmt:
		y, ok := y.(*ast.LabeledStmt)
		return ok && astLabeledStmtEq(x, y)

	case *ast.EmptyStmt:
		y, ok := y.(*ast.EmptyStmt)
		return ok && astEmptyStmtEq(x, y)

	default:
		return false
	}
}

func astDeclEq(x, y ast.Decl) bool {
	if x == nil || y == nil {
		return x == y
	}

	switch x := x.(type) {
	case *ast.GenDecl:
		y, ok := y.(*ast.GenDecl)
		return ok && astGenDeclEq(x, y)

	case *ast.FuncDecl:
		y, ok := y.(*ast.FuncDecl)
		return ok && astFuncDeclEq(x, y)

	default:
		return false
	}
}

// Compare concrete nodes for equallity.
//
// Any node of pointer type permitted to be nil,
// hence nil checks are mandatory.

func astIdentEq(x, y *ast.Ident) bool {
	if x == nil || y == nil {
		return x == y
	}
	return x.Name == y.Name
}

func astKeyValueExprEq(x, y *ast.KeyValueExpr) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.Key, y.Key) && astExprEq(x.Value, y.Value)
}

func astArrayTypeEq(x, y *ast.ArrayType) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.Len, y.Len) && astExprEq(x.Elt, y.Elt)
}

func astStructTypeEq(x, y *ast.StructType) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astFieldListEq(x.Fields, y.Fields)
}

func astFuncTypeEq(x, y *ast.FuncType) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astFieldListEq(x.Params, y.Params) &&
		astFieldListEq(x.Results, y.Results)
}

func astBasicLitEq(x, y *ast.BasicLit) bool {
	if x == nil || y == nil {
		return x == y
	}
	return x.Kind == y.Kind && x.Value == y.Value
}

func astBlockStmtEq(x, y *ast.BlockStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astStmtSliceEq(x.List, y.List)
}

func astFieldEq(x, y *ast.Field) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astIdentSliceEq(x.Names, y.Names) &&
		astExprEq(x.Type, y.Type)
}

func astFuncLitEq(x, y *ast.FuncLit) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astFuncTypeEq(x.Type, y.Type) &&
		astBlockStmtEq(x.Body, y.Body)
}

func astCompositeLitEq(x, y *ast.CompositeLit) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.Type, y.Type) &&
		astExprSliceEq(x.Elts, y.Elts)
}

func astSelectorExprEq(x, y *ast.SelectorExpr) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.X, y.X) && astIdentEq(x.Sel, y.Sel)
}

func astIndexExprEq(x, y *ast.IndexExpr) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.X, y.X) && astExprEq(x.Index, y.Index)
}

func astSliceExprEq(x, y *ast.SliceExpr) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.X, y.X) &&
		astExprEq(x.Low, y.Low) &&
		astExprEq(x.High, y.High) &&
		astExprEq(x.Max, y.Max)
}

func astTypeAssertExprEq(x, y *ast.TypeAssertExpr) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.X, y.X) && astExprEq(x.Type, y.Type)
}

func astInterfaceTypeEq(x, y *ast.InterfaceType) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astFieldListEq(x.Methods, y.Methods)
}

func astMapTypeEq(x, y *ast.MapType) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.Key, y.Key) && astExprEq(x.Value, y.Value)
}

func astChanTypeEq(x, y *ast.ChanType) bool {
	if x == nil || y == nil {
		return x == y
	}
	return x.Dir == y.Dir && astExprEq(x.Value, y.Value)
}

func astCallExprEq(x, y *ast.CallExpr) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.Fun, y.Fun) &&
		astExprSliceEq(x.Args, y.Args) &&
		(x.Ellipsis == 0) == (y.Ellipsis == 0)
}

func astEllipsisEq(x, y *ast.Ellipsis) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.Elt, y.Elt)
}

func astUnaryExprEq(x, y *ast.UnaryExpr) bool {
	if x == nil || y == nil {
		return x == y
	}
	return x.Op == y.Op && astExprEq(x.X, y.X)
}

func astBinaryExprEq(x, y *ast.BinaryExpr) bool {
	if x == nil || y == nil {
		return x == y
	}
	return x.Op == y.Op &&
		astExprEq(x.X, y.X) &&
		astExprEq(x.Y, y.Y)
}

func astParenExprEq(x, y *ast.ParenExpr) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.X, y.X)
}

func astStarExprEq(x, y *ast.StarExpr) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.X, y.X)
}

func astFieldListEq(x, y *ast.FieldList) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astFieldSliceEq(x.List, y.List)
}

func astEmptyStmtEq(x, y *ast.EmptyStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return x.Implicit == y.Implicit
}

func astLabeledStmtEq(x, y *ast.LabeledStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astIdentEq(x.Label, y.Label) && astStmtEq(x.Stmt, y.Stmt)
}

func astExprStmtEq(x, y *ast.ExprStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.X, y.X)
}

func astSendStmtEq(x, y *ast.SendStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprEq(x.Chan, y.Chan) && astExprEq(x.Value, y.Value)
}

func astDeclStmtEq(x, y *ast.DeclStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astDeclEq(x.Decl, y.Decl)
}

func astIncDecStmtEq(x, y *ast.IncDecStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return x.Tok == y.Tok && astExprEq(x.X, y.X)
}

func astAssignStmtEq(x, y *ast.AssignStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return x.Tok == y.Tok &&
		astExprSliceEq(x.Lhs, y.Lhs) &&
		astExprSliceEq(x.Rhs, y.Rhs)
}

func astGoStmtEq(x, y *ast.GoStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astCallExprEq(x.Call, y.Call)
}

func astDeferStmtEq(x, y *ast.DeferStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astCallExprEq(x.Call, y.Call)
}

func astReturnStmtEq(x, y *ast.ReturnStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprSliceEq(x.Results, y.Results)
}

func astBranchStmtEq(x, y *ast.BranchStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return x.Tok == y.Tok && astIdentEq(x.Label, y.Label)
}

func astIfStmtEq(x, y *ast.IfStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astStmtEq(x.Init, y.Init) &&
		astExprEq(x.Cond, y.Cond) &&
		astBlockStmtEq(x.Body, y.Body) &&
		astStmtEq(x.Else, y.Else)
}

func astCaseClauseEq(x, y *ast.CaseClause) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astExprSliceEq(x.List, y.List) &&
		astStmtSliceEq(x.Body, y.Body)
}

func astSwitchStmtEq(x, y *ast.SwitchStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astStmtEq(x.Init, y.Init) &&
		astExprEq(x.Tag, y.Tag) &&
		astBlockStmtEq(x.Body, y.Body)
}

func astTypeSwitchStmtEq(x, y *ast.TypeSwitchStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astStmtEq(x.Init, y.Init) &&
		astStmtEq(x.Assign, y.Assign) &&
		astBlockStmtEq(x.Body, y.Body)
}

func astCommClauseEq(x, y *ast.CommClause) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astStmtEq(x.Comm, y.Comm) && astStmtSliceEq(x.Body, y.Body)
}

func astSelectStmtEq(x, y *ast.SelectStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astBlockStmtEq(x.Body, y.Body)
}

func astForStmtEq(x, y *ast.ForStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astStmtEq(x.Init, y.Init) &&
		astExprEq(x.Cond, y.Cond) &&
		astStmtEq(x.Post, y.Post) &&
		astBlockStmtEq(x.Body, y.Body)
}

func astRangeStmtEq(x, y *ast.RangeStmt) bool {
	if x == nil || y == nil {
		return x == y
	}
	return x.Tok == y.Tok &&
		astExprEq(x.Key, y.Key) &&
		astExprEq(x.Value, y.Value) &&
		astExprEq(x.X, y.X) &&
		astBlockStmtEq(x.Body, y.Body)
}

func astFuncDeclEq(x, y *ast.FuncDecl) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astFieldListEq(x.Recv, y.Recv) &&
		astIdentEq(x.Name, y.Name) &&
		astFuncTypeEq(x.Type, y.Type) &&
		astBlockStmtEq(x.Body, y.Body)
}

func astGenDeclEq(x, y *ast.GenDecl) bool {
	if x == nil || y == nil {
		return x == y
	}

	if x.Tok != y.Tok {
		return false
	}
	if len(x.Specs) != len(y.Specs) {
		return false
	}

	switch x.Tok {
	case token.IMPORT:
		for i := range x.Specs {
			xspec := x.Specs[i].(*ast.ImportSpec)
			yspec := y.Specs[i].(*ast.ImportSpec)
			if !astImportSpecEq(xspec, yspec) {
				return false
			}
		}
	case token.TYPE:
		for i := range x.Specs {
			xspec := x.Specs[i].(*ast.TypeSpec)
			yspec := y.Specs[i].(*ast.TypeSpec)
			if !astTypeSpecEq(xspec, yspec) {
				return false
			}
		}
	default:
		for i := range x.Specs {
			xspec := x.Specs[i].(*ast.ValueSpec)
			yspec := y.Specs[i].(*ast.ValueSpec)
			if !astValueSpecEq(xspec, yspec) {
				return false
			}
		}
	}

	return true
}

func astImportSpecEq(x, y *ast.ImportSpec) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astIdentEq(x.Name, y.Name) && astBasicLitEq(x.Path, y.Path)
}

func astTypeSpecEq(x, y *ast.TypeSpec) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astIdentEq(x.Name, y.Name) && astExprEq(x.Type, y.Type)
}

func astValueSpecEq(x, y *ast.ValueSpec) bool {
	if x == nil || y == nil {
		return x == y
	}
	return astIdentSliceEq(x.Names, y.Names) &&
		astExprEq(x.Type, y.Type) &&
		astExprSliceEq(x.Values, y.Values)
}

// Compare slices for equallity.
//
// Each slice element that has pointer type permitted to be nil,
// hence instead of using adhoc comparison of values,
// equallity functions that are defined above are used.

func astIdentSliceEq(xs, ys []*ast.Ident) bool {
	if len(xs) != len(ys) {
		return false
	}
	for i := range xs {
		if !astIdentEq(xs[i], ys[i]) {
			return false
		}
	}
	return true
}

func astFieldSliceEq(xs, ys []*ast.Field) bool {
	if len(xs) != len(ys) {
		return false
	}
	for i := range xs {
		if !astFieldEq(xs[i], ys[i]) {
			return false
		}
	}
	return true
}

func astStmtSliceEq(xs, ys []ast.Stmt) bool {
	if len(xs) != len(ys) {
		return false
	}
	for i := range xs {
		if !astStmtEq(xs[i], ys[i]) {
			return false
		}
	}
	return true
}

func astExprSliceEq(xs, ys []ast.Expr) bool {
	if len(xs) != len(ys) {
		return false
	}
	for i := range xs {
		if !astExprEq(xs[i], ys[i]) {
			return false
		}
	}
	return true
}
