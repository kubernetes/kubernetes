package ruleguard

import (
	"fmt"
	"go/ast"
	"strings"
)

type nodePath struct {
	stack []ast.Node
}

func newNodePath() nodePath {
	return nodePath{stack: make([]ast.Node, 0, 32)}
}

func (p nodePath) String() string {
	parts := make([]string, len(p.stack))
	for i, n := range p.stack {
		parts[i] = fmt.Sprintf("%T", n)
	}
	return strings.Join(parts, "/")
}

func (p nodePath) Parent() ast.Node {
	return p.NthParent(1)
}

func (p nodePath) Current() ast.Node {
	return p.NthParent(0)
}

func (p nodePath) NthParent(n int) ast.Node {
	index := len(p.stack) - n - 1
	if index >= 0 {
		return p.stack[index]
	}
	return nil
}

func (p *nodePath) Len() int { return len(p.stack) }

func (p *nodePath) Push(n ast.Node) {
	p.stack = append(p.stack, n)
}

func (p *nodePath) Pop() {
	p.stack = p.stack[:len(p.stack)-1]
}
