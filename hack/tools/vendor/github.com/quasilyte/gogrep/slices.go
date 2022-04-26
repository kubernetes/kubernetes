package gogrep

import (
	"go/ast"
	"go/token"
)

type NodeSlice interface {
	At(i int) ast.Node
	Len() int
	slice(from, to int) NodeSlice
	ast.Node
}

type (
	ExprSlice  []ast.Expr
	stmtSlice  []ast.Stmt
	fieldSlice []*ast.Field
	identSlice []*ast.Ident
	specSlice  []ast.Spec
	declSlice  []ast.Decl
)

func (l ExprSlice) Len() int                 { return len(l) }
func (l ExprSlice) At(i int) ast.Node        { return l[i] }
func (l ExprSlice) slice(i, j int) NodeSlice { return l[i:j] }
func (l ExprSlice) Pos() token.Pos           { return l[0].Pos() }
func (l ExprSlice) End() token.Pos           { return l[len(l)-1].End() }

func (l stmtSlice) Len() int                 { return len(l) }
func (l stmtSlice) At(i int) ast.Node        { return l[i] }
func (l stmtSlice) slice(i, j int) NodeSlice { return l[i:j] }
func (l stmtSlice) Pos() token.Pos           { return l[0].Pos() }
func (l stmtSlice) End() token.Pos           { return l[len(l)-1].End() }

func (l fieldSlice) Len() int                 { return len(l) }
func (l fieldSlice) At(i int) ast.Node        { return l[i] }
func (l fieldSlice) slice(i, j int) NodeSlice { return l[i:j] }
func (l fieldSlice) Pos() token.Pos           { return l[0].Pos() }
func (l fieldSlice) End() token.Pos           { return l[len(l)-1].End() }

func (l identSlice) Len() int                 { return len(l) }
func (l identSlice) At(i int) ast.Node        { return l[i] }
func (l identSlice) slice(i, j int) NodeSlice { return l[i:j] }
func (l identSlice) Pos() token.Pos           { return l[0].Pos() }
func (l identSlice) End() token.Pos           { return l[len(l)-1].End() }

func (l specSlice) Len() int                 { return len(l) }
func (l specSlice) At(i int) ast.Node        { return l[i] }
func (l specSlice) slice(i, j int) NodeSlice { return l[i:j] }
func (l specSlice) Pos() token.Pos           { return l[0].Pos() }
func (l specSlice) End() token.Pos           { return l[len(l)-1].End() }

func (l declSlice) Len() int                 { return len(l) }
func (l declSlice) At(i int) ast.Node        { return l[i] }
func (l declSlice) slice(i, j int) NodeSlice { return l[i:j] }
func (l declSlice) Pos() token.Pos           { return l[0].Pos() }
func (l declSlice) End() token.Pos           { return l[len(l)-1].End() }
