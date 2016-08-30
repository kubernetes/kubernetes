package ast

import (
	"bytes"
	"fmt"
)

// Arithmetic represents a node where the result is arithmetic of
// two or more operands in the order given.
type Arithmetic struct {
	Op    ArithmeticOp
	Exprs []Node
	Posx  Pos
}

func (n *Arithmetic) Accept(v Visitor) Node {
	for i, expr := range n.Exprs {
		n.Exprs[i] = expr.Accept(v)
	}

	return v(n)
}

func (n *Arithmetic) Pos() Pos {
	return n.Posx
}

func (n *Arithmetic) GoString() string {
	return fmt.Sprintf("*%#v", *n)
}

func (n *Arithmetic) String() string {
	var b bytes.Buffer
	for _, expr := range n.Exprs {
		b.WriteString(fmt.Sprintf("%s", expr))
	}

	return b.String()
}

func (n *Arithmetic) Type(Scope) (Type, error) {
	return TypeInt, nil
}
