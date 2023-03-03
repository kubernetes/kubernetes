// Package treeprint provides a simple ASCII tree composing tool.
package treeprint

import (
	"bytes"
	"fmt"
	"io"
	"reflect"
	"strings"
)

// Value defines any value
type Value interface{}

// MetaValue defines any meta value
type MetaValue interface{}

// NodeVisitor function type for iterating over nodes
type NodeVisitor func(item *node)

// Tree represents a tree structure with leaf-nodes and branch-nodes.
type Tree interface {
	// AddNode adds a new node to a branch.
	AddNode(v Value) Tree
	// AddMetaNode adds a new node with meta value provided to a branch.
	AddMetaNode(meta MetaValue, v Value) Tree
	// AddBranch adds a new branch node (a level deeper).
	AddBranch(v Value) Tree
	// AddMetaBranch adds a new branch node (a level deeper) with meta value provided.
	AddMetaBranch(meta MetaValue, v Value) Tree
	// Branch converts a leaf-node to a branch-node,
	// applying this on a branch-node does no effect.
	Branch() Tree
	// FindByMeta finds a node whose meta value matches the provided one by reflect.DeepEqual,
	// returns nil if not found.
	FindByMeta(meta MetaValue) Tree
	// FindByValue finds a node whose value matches the provided one by reflect.DeepEqual,
	// returns nil if not found.
	FindByValue(value Value) Tree
	//  returns the last node of a tree
	FindLastNode() Tree
	// String renders the tree or subtree as a string.
	String() string
	// Bytes renders the tree or subtree as byteslice.
	Bytes() []byte

	SetValue(value Value)
	SetMetaValue(meta MetaValue)

	// VisitAll iterates over the tree, branches and nodes.
	// If need to iterate over the whole tree, use the root node.
	// Note this method uses a breadth-first approach.
	VisitAll(fn NodeVisitor)
}

type node struct {
	Root  *node
	Meta  MetaValue
	Value Value
	Nodes []*node
}

func (n *node) FindLastNode() Tree {
	ns := n.Nodes
	if len(ns) == 0 {
		return nil
	}
	return ns[len(ns)-1]
}

func (n *node) AddNode(v Value) Tree {
	n.Nodes = append(n.Nodes, &node{
		Root:  n,
		Value: v,
	})
	return n
}

func (n *node) AddMetaNode(meta MetaValue, v Value) Tree {
	n.Nodes = append(n.Nodes, &node{
		Root:  n,
		Meta:  meta,
		Value: v,
	})
	return n
}

func (n *node) AddBranch(v Value) Tree {
	branch := &node{
		Root:  n,
		Value: v,
	}
	n.Nodes = append(n.Nodes, branch)
	return branch
}

func (n *node) AddMetaBranch(meta MetaValue, v Value) Tree {
	branch := &node{
		Root:  n,
		Meta:  meta,
		Value: v,
	}
	n.Nodes = append(n.Nodes, branch)
	return branch
}

func (n *node) Branch() Tree {
	n.Root = nil
	return n
}

func (n *node) FindByMeta(meta MetaValue) Tree {
	for _, node := range n.Nodes {
		if reflect.DeepEqual(node.Meta, meta) {
			return node
		}
		if v := node.FindByMeta(meta); v != nil {
			return v
		}
	}
	return nil
}

func (n *node) FindByValue(value Value) Tree {
	for _, node := range n.Nodes {
		if reflect.DeepEqual(node.Value, value) {
			return node
		}
		if v := node.FindByMeta(value); v != nil {
			return v
		}
	}
	return nil
}

func (n *node) Bytes() []byte {
	buf := new(bytes.Buffer)
	level := 0
	var levelsEnded []int
	if n.Root == nil {
		if n.Meta != nil {
			buf.WriteString(fmt.Sprintf("[%v]  %v", n.Meta, n.Value))
		} else {
			buf.WriteString(fmt.Sprintf("%v", n.Value))
		}
		buf.WriteByte('\n')
	} else {
		edge := EdgeTypeMid
		if len(n.Nodes) == 0 {
			edge = EdgeTypeEnd
			levelsEnded = append(levelsEnded, level)
		}
		printValues(buf, 0, levelsEnded, edge, n)
	}
	if len(n.Nodes) > 0 {
		printNodes(buf, level, levelsEnded, n.Nodes)
	}
	return buf.Bytes()
}

func (n *node) String() string {
	return string(n.Bytes())
}

func (n *node) SetValue(value Value) {
	n.Value = value
}

func (n *node) SetMetaValue(meta MetaValue) {
	n.Meta = meta
}

func (n *node) VisitAll(fn NodeVisitor) {
	for _, node := range n.Nodes {
		fn(node)

		if len(node.Nodes) > 0 {
			node.VisitAll(fn)
			continue
		}
	}
}

func printNodes(wr io.Writer,
	level int, levelsEnded []int, nodes []*node) {

	for i, node := range nodes {
		edge := EdgeTypeMid
		if i == len(nodes)-1 {
			levelsEnded = append(levelsEnded, level)
			edge = EdgeTypeEnd
		}
		printValues(wr, level, levelsEnded, edge, node)
		if len(node.Nodes) > 0 {
			printNodes(wr, level+1, levelsEnded, node.Nodes)
		}
	}
}

func printValues(wr io.Writer,
	level int, levelsEnded []int, edge EdgeType, node *node) {

	for i := 0; i < level; i++ {
		if isEnded(levelsEnded, i) {
			fmt.Fprint(wr, strings.Repeat(" ", IndentSize+1))
			continue
		}
		fmt.Fprintf(wr, "%s%s", EdgeTypeLink, strings.Repeat(" ", IndentSize))
	}

	val := renderValue(level, node)
	meta := node.Meta

	if meta != nil {
		fmt.Fprintf(wr, "%s [%v]  %v\n", edge, meta, val)
		return
	}
	fmt.Fprintf(wr, "%s %v\n", edge, val)
}

func isEnded(levelsEnded []int, level int) bool {
	for _, l := range levelsEnded {
		if l == level {
			return true
		}
	}
	return false
}

func renderValue(level int, node *node) Value {
	lines := strings.Split(fmt.Sprintf("%v", node.Value), "\n")

	// If value does not contain multiple lines, return itself.
	if len(lines) < 2 {
		return node.Value
	}

	// If value contains multiple lines,
	// generate a padding and prefix each line with it.
	pad := padding(level, node)

	for i := 1; i < len(lines); i++ {
		lines[i] = fmt.Sprintf("%s%s", pad, lines[i])
	}

	return strings.Join(lines, "\n")
}

// padding returns a padding for the multiline values with correctly placed link edges.
// It is generated by traversing the tree upwards (from leaf to the root of the tree)
// and, on each level, checking if the node the last one of its siblings.
// If a node is the last one, the padding on that level should be empty (there's nothing to link to below it).
// If a node is not the last one, the padding on that level should be the link edge so the sibling below is correctly connected.
func padding(level int, node *node) string {
	links := make([]string, level+1)

	for node.Root != nil {
		if isLast(node) {
			links[level] = strings.Repeat(" ", IndentSize+1)
		} else {
			links[level] = fmt.Sprintf("%s%s", EdgeTypeLink, strings.Repeat(" ", IndentSize))
		}
		level--
		node = node.Root
	}

	return strings.Join(links, "")
}

// isLast checks if the node is the last one in the slice of its parent children
func isLast(n *node) bool {
	return n == n.Root.FindLastNode()
}

type EdgeType string

var (
	EdgeTypeLink EdgeType = "│"
	EdgeTypeMid  EdgeType = "├──"
	EdgeTypeEnd  EdgeType = "└──"
)

// IndentSize is the number of spaces per tree level.
var IndentSize = 3

// New Generates new tree
func New() Tree {
	return &node{Value: "."}
}

// NewWithRoot Generates new tree with the given root value
func NewWithRoot(root Value) Tree {
	return &node{Value: root}
}
