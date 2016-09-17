package hil

import (
	"fmt"
	"sync"

	"github.com/hashicorp/hil/ast"
)

// IdentifierCheck is a SemanticCheck that checks that all identifiers
// resolve properly and that the right number of arguments are passed
// to functions.
type IdentifierCheck struct {
	Scope ast.Scope

	err  error
	lock sync.Mutex
}

func (c *IdentifierCheck) Visit(root ast.Node) error {
	c.lock.Lock()
	defer c.lock.Unlock()
	defer c.reset()
	root.Accept(c.visit)
	return c.err
}

func (c *IdentifierCheck) visit(raw ast.Node) ast.Node {
	if c.err != nil {
		return raw
	}

	switch n := raw.(type) {
	case *ast.Call:
		c.visitCall(n)
	case *ast.VariableAccess:
		c.visitVariableAccess(n)
	case *ast.Output:
		// Ignore
	case *ast.LiteralNode:
		// Ignore
	default:
		// Ignore
	}

	// We never do replacement with this visitor
	return raw
}

func (c *IdentifierCheck) visitCall(n *ast.Call) {
	// Look up the function in the map
	function, ok := c.Scope.LookupFunc(n.Func)
	if !ok {
		c.createErr(n, fmt.Sprintf("unknown function called: %s", n.Func))
		return
	}

	// Break up the args into what is variadic and what is required
	args := n.Args
	if function.Variadic && len(args) > len(function.ArgTypes) {
		args = n.Args[:len(function.ArgTypes)]
	}

	// Verify the number of arguments
	if len(args) != len(function.ArgTypes) {
		c.createErr(n, fmt.Sprintf(
			"%s: expected %d arguments, got %d",
			n.Func, len(function.ArgTypes), len(n.Args)))
		return
	}
}

func (c *IdentifierCheck) visitVariableAccess(n *ast.VariableAccess) {
	// Look up the variable in the map
	if _, ok := c.Scope.LookupVar(n.Name); !ok {
		c.createErr(n, fmt.Sprintf(
			"unknown variable accessed: %s", n.Name))
		return
	}
}

func (c *IdentifierCheck) createErr(n ast.Node, str string) {
	c.err = fmt.Errorf("%s: %s", n.Pos(), str)
}

func (c *IdentifierCheck) reset() {
	c.err = nil
}
