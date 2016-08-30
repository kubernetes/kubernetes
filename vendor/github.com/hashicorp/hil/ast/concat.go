package ast

import (
	"bytes"
	"fmt"
)

// Concat represents a node where the result of two or more expressions are
// concatenated. The result of all expressions must be a string.
type Concat struct {
	Exprs []Node
	Posx  Pos
}

func (n *Concat) Accept(v Visitor) Node {
	for i, expr := range n.Exprs {
		n.Exprs[i] = expr.Accept(v)
	}

	return v(n)
}

func (n *Concat) Pos() Pos {
	return n.Posx
}

func (n *Concat) GoString() string {
	return fmt.Sprintf("*%#v", *n)
}

func (n *Concat) String() string {
	var b bytes.Buffer
	for _, expr := range n.Exprs {
		b.WriteString(fmt.Sprintf("%s", expr))
	}

	return b.String()
}

func (n *Concat) Type(Scope) (Type, error) {
	return TypeString, nil
}
