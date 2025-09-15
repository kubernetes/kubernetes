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
type NodeVisitor func(item *Node)

// Tree represents a tree structure with leaf-nodes and branch-nodes.
type Tree interface {
	// AddNode adds a new Node to a branch.
	AddNode(v Value) Tree
	// AddMetaNode adds a new Node with meta value provided to a branch.
	AddMetaNode(meta MetaValue, v Value) Tree
	// AddBranch adds a new branch Node (a level deeper).
	AddBranch(v Value) Tree
	// AddMetaBranch adds a new branch Node (a level deeper) with meta value provided.
	AddMetaBranch(meta MetaValue, v Value) Tree
	// Branch converts a leaf-Node to a branch-Node,
	// applying this on a branch-Node does no effect.
	Branch() Tree
	// FindByMeta finds a Node whose meta value matches the provided one by reflect.DeepEqual,
	// returns nil if not found.
	FindByMeta(meta MetaValue) Tree
	// FindByValue finds a Node whose value matches the provided one by reflect.DeepEqual,
	// returns nil if not found.
	FindByValue(value Value) Tree
	//  returns the last Node of a tree
	FindLastNode() Tree
	// String renders the tree or subtree as a string.
	String() string
	// Bytes renders the tree or subtree as byteslice.
	Bytes() []byte

	SetValue(value Value)
	SetMetaValue(meta MetaValue)

	// VisitAll iterates over the tree, branches and nodes.
	// If need to iterate over the whole tree, use the root Node.
	// Note this method uses a breadth-first approach.
	VisitAll(fn NodeVisitor)
}

type Node struct {
	Root  *Node
	Meta  MetaValue
	Value Value
	Nodes []*Node
}

func (n *Node) FindLastNode() Tree {
	ns := n.Nodes
	if len(ns) == 0 {
		return nil
	}
	return ns[len(ns)-1]
}

func (n *Node) AddNode(v Value) Tree {
	n.Nodes = append(n.Nodes, &Node{
		Root:  n,
		Value: v,
	})
	return n
}

func (n *Node) AddMetaNode(meta MetaValue, v Value) Tree {
	n.Nodes = append(n.Nodes, &Node{
		Root:  n,
		Meta:  meta,
		Value: v,
	})
	return n
}

func (n *Node) AddBranch(v Value) Tree {
	branch := &Node{
		Root:  n,
		Value: v,
	}
	n.Nodes = append(n.Nodes, branch)
	return branch
}

func (n *Node) AddMetaBranch(meta MetaValue, v Value) Tree {
	branch := &Node{
		Root:  n,
		Meta:  meta,
		Value: v,
	}
	n.Nodes = append(n.Nodes, branch)
	return branch
}

func (n *Node) Branch() Tree {
	n.Root = nil
	return n
}

func (n *Node) FindByMeta(meta MetaValue) Tree {
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

func (n *Node) FindByValue(value Value) Tree {
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

func (n *Node) Bytes() []byte {
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

func (n *Node) String() string {
	return string(n.Bytes())
}

func (n *Node) SetValue(value Value) {
	n.Value = value
}

func (n *Node) SetMetaValue(meta MetaValue) {
	n.Meta = meta
}

func (n *Node) VisitAll(fn NodeVisitor) {
	for _, node := range n.Nodes {
		fn(node)

		if len(node.Nodes) > 0 {
			node.VisitAll(fn)
			continue
		}
	}
}

func printNodes(wr io.Writer,
	level int, levelsEnded []int, nodes []*Node) {

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
	level int, levelsEnded []int, edge EdgeType, node *Node) {

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

func renderValue(level int, node *Node) Value {
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
// and, on each level, checking if the Node the last one of its siblings.
// If a Node is the last one, the padding on that level should be empty (there's nothing to link to below it).
// If a Node is not the last one, the padding on that level should be the link edge so the sibling below is correctly connected.
func padding(level int, node *Node) string {
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

// isLast checks if the Node is the last one in the slice of its parent children
func isLast(n *Node) bool {
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
	return &Node{Value: "."}
}

// NewWithRoot Generates new tree with the given root value
func NewWithRoot(root Value) Tree {
	return &Node{Value: root}
}
