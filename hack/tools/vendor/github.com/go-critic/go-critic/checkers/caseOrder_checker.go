package checkers

import (
	"go/ast"
	"go/types"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "caseOrder"
	info.Tags = []string{"diagnostic"}
	info.Summary = "Detects erroneous case order inside switch statements"
	info.Before = `
switch x.(type) {
case ast.Expr:
	fmt.Println("expr")
case *ast.BasicLit:
	fmt.Println("basic lit") // Never executed
}`
	info.After = `
switch x.(type) {
case *ast.BasicLit:
	fmt.Println("basic lit") // Now reachable
case ast.Expr:
	fmt.Println("expr")
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&caseOrderChecker{ctx: ctx}), nil
	})
}

type caseOrderChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *caseOrderChecker) VisitStmt(stmt ast.Stmt) {
	switch stmt := stmt.(type) {
	case *ast.TypeSwitchStmt:
		c.checkTypeSwitch(stmt)
	case *ast.SwitchStmt:
		c.checkSwitch(stmt)
	}
}

func (c *caseOrderChecker) checkTypeSwitch(s *ast.TypeSwitchStmt) {
	type ifaceType struct {
		node ast.Node
		typ  *types.Interface
	}
	var ifaces []ifaceType // Interfaces seen so far
	for _, cc := range s.Body.List {
		cc := cc.(*ast.CaseClause)
		for _, x := range cc.List {
			typ := c.ctx.TypeOf(x)
			if typ == linter.UnknownType {
				c.warnUnknownType(cc, x)
				return
			}
			for _, iface := range ifaces {
				if types.Implements(typ, iface.typ) {
					c.warnTypeSwitch(cc, x, iface.node)
					break
				}
			}
			if iface, ok := typ.Underlying().(*types.Interface); ok {
				ifaces = append(ifaces, ifaceType{node: x, typ: iface})
			}
		}
	}
}

func (c *caseOrderChecker) warnTypeSwitch(cause, concrete, iface ast.Node) {
	c.ctx.Warn(cause, "case %s must go before the %s case", concrete, iface)
}

func (c *caseOrderChecker) warnUnknownType(cause, concrete ast.Node) {
	c.ctx.Warn(cause, "type is not defined %s", concrete)
}

func (c *caseOrderChecker) checkSwitch(s *ast.SwitchStmt) {
	// TODO(quasilyte): can handle expression cases that overlap.
	// Cases that have narrower value range should go before wider ones.
}
