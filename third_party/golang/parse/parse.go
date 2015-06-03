package parse

import (
	"bytes"
	"fmt"
)

var textFormat = "%s" // Changed to "%q" in tests for better error messages.

type Node interface {
	Type() NodeType
	Copy() Node
}

// NodeType identifies the type of a parse tree node.
type NodeType int

func (p Pos) Position() Pos {
	return p
}

// unexported keeps Node implementations local to the package.
// All implementations embed Pos, so this takes care of it.
func (Pos) unexported() {
}

// Type returns itself and provides an easy default implementation
// for embedding in a Node. Embedded in all non-trivial Nodes.
func (t NodeType) Type() NodeType {
	return t
}

const (
	NodeText NodeType = iota
	NodeList
	NodeVariable
)

// ListNode holds a sequence of nodes.
type ListNode struct {
	NodeType
	Pos
	Nodes []Node // The element nodes in lexical order.
}

func newList(pos Pos) *ListNode {
	return &ListNode{NodeType: NodeList, Pos: pos}
}

func (l *ListNode) Type() NodeType {
	return l.NodeType
}

func (l *ListNode) append(n Node) {
	l.Nodes = append(l.Nodes, n)
}

func (l *ListNode) String() string {
	b := new(bytes.Buffer)
	for _, n := range l.Nodes {
		fmt.Fprint(b, n)
	}
	return b.String()
}

func (l *ListNode) CopyList() *ListNode {
	if l == nil {
		return l
	}
	n := newList(l.Pos)
	for _, elem := range l.Nodes {
		n.append(elem.Copy())
	}
	return n
}

func (l *ListNode) Copy() Node {
	return l.CopyList()
}

// TextNode holds plain text.
type TextNode struct {
	NodeType
	Pos
	Text []byte // The text; may span newlines.
}

func newText(pos Pos, text string) *TextNode {
	return &TextNode{NodeType: NodeText, Pos: pos, Text: []byte(text)}
}

func (t *TextNode) String() string {
	return fmt.Sprintf(textFormat, t.Text)
}

func (t *TextNode) Copy() Node {
	return &TextNode{NodeType: NodeText, Text: append([]byte{}, t.Text...)}
}

type VariableNode struct {
	Name string
	NodeType
	Pos
}

func newVariable(pos Pos, name string) *VariableNode {
	return &VariableNode{NodeType: NodeVariable, Pos: pos, Name: name}
}

func (v *VariableNode) Copy() Node {
	return &VariableNode{NodeType: NodeVariable, Pos: v.Pos, Name: v.Name}
}

type Tree struct {
	Name string
	lex  *lexer
	Root *ListNode
	text string
	cur  *ListNode
}

// Parse parsed the given text and return a node Tree.
// If an error is encountered, parsing stops and an empty
// Tree is returned with the error
func Parse(name, text string) (*Tree, error) {
	t := NewTree(name)
	return t, t.Parse(text)
}

func NewTree(name string) *Tree {
	return &Tree{
		Name: name,
	}
}

func (t *Tree) Parse(text string) error {
	t.Root = newList(0)
	t.lex = lex(t.Name, text, "${", "}")
	t.cur = t.Root
	for eof := false; eof != true; {
		item := t.lex.nextItem()
		switch item.typ {
		case itemLeftDelim:
			newNode := newList(item.pos)
			t.cur.append(newNode)
			t.cur = newNode
		case itemRightDelim:
			t.cur = t.Root
		case itemField:
			newNode := newVariable(item.pos, item.val)
			t.cur.append(newNode)
		case itemText:
			t.cur.append(newText(item.pos, item.val))
		case itemString:
			t.cur.append(newText(item.pos, item.val[1:len(item.val)-1]))
		case itemEOF:
			eof = true
		}
	}
	return nil
}
