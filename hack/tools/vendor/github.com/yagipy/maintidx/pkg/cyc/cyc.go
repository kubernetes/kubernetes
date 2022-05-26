package cyc

import (
	"go/ast"
	"go/token"
)

type Cyc struct {
	Val  int
	Coef Coef
}

type Coef struct{}

func (c *Cyc) Analyze(n ast.Node) {
	switch n := n.(type) {
	case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt:
		c.Val++
	case *ast.CaseClause:
		if n.List != nil {
			c.Val++
		}
	case *ast.CommClause:
		if n.Comm != nil {
			c.Val++
		}
	case *ast.BinaryExpr:
		if n.Op == token.LAND || n.Op == token.LOR {
			c.Val++
		}
	}
}

// TODO: Implement
func (c *Cyc) Calc() {
}
