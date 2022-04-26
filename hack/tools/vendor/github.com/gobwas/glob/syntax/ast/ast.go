package ast

import (
	"bytes"
	"fmt"
)

type Node struct {
	Parent   *Node
	Children []*Node
	Value    interface{}
	Kind     Kind
}

func NewNode(k Kind, v interface{}, ch ...*Node) *Node {
	n := &Node{
		Kind:  k,
		Value: v,
	}
	for _, c := range ch {
		Insert(n, c)
	}
	return n
}

func (a *Node) Equal(b *Node) bool {
	if a.Kind != b.Kind {
		return false
	}
	if a.Value != b.Value {
		return false
	}
	if len(a.Children) != len(b.Children) {
		return false
	}
	for i, c := range a.Children {
		if !c.Equal(b.Children[i]) {
			return false
		}
	}
	return true
}

func (a *Node) String() string {
	var buf bytes.Buffer
	buf.WriteString(a.Kind.String())
	if a.Value != nil {
		buf.WriteString(" =")
		buf.WriteString(fmt.Sprintf("%v", a.Value))
	}
	if len(a.Children) > 0 {
		buf.WriteString(" [")
		for i, c := range a.Children {
			if i > 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(c.String())
		}
		buf.WriteString("]")
	}
	return buf.String()
}

func Insert(parent *Node, children ...*Node) {
	parent.Children = append(parent.Children, children...)
	for _, ch := range children {
		ch.Parent = parent
	}
}

type List struct {
	Not   bool
	Chars string
}

type Range struct {
	Not    bool
	Lo, Hi rune
}

type Text struct {
	Text string
}

type Kind int

const (
	KindNothing Kind = iota
	KindPattern
	KindList
	KindRange
	KindText
	KindAny
	KindSuper
	KindSingle
	KindAnyOf
)

func (k Kind) String() string {
	switch k {
	case KindNothing:
		return "Nothing"
	case KindPattern:
		return "Pattern"
	case KindList:
		return "List"
	case KindRange:
		return "Range"
	case KindText:
		return "Text"
	case KindAny:
		return "Any"
	case KindSuper:
		return "Super"
	case KindSingle:
		return "Single"
	case KindAnyOf:
		return "AnyOf"
	default:
		return ""
	}
}
