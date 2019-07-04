// This file is dual licensed under CC0 and The gonum license.
//
// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Copyright ©2017 Robin Eklind.
// This file is made available under a Creative Commons CC0 1.0
// Universal Public Domain Dedication.

package ast

import (
	"bytes"
	"fmt"
)

// === [ File ] ================================================================

// A File represents a DOT file.
//
// Examples.
//
//    digraph G {
//       A -> B
//    }
//    graph H {
//       C - D
//    }
type File struct {
	// Graphs.
	Graphs []*Graph
}

// String returns the string representation of the file.
func (f *File) String() string {
	buf := new(bytes.Buffer)
	for i, graph := range f.Graphs {
		if i != 0 {
			buf.WriteString("\n")
		}
		buf.WriteString(graph.String())
	}
	return buf.String()
}

// === [ Graphs ] ==============================================================

// A Graph represents a directed or an undirected graph.
//
// Examples.
//
//    digraph G {
//       A -> {B C}
//       B -> C
//    }
type Graph struct {
	// Strict graph; multi-edges forbidden.
	Strict bool
	// Directed graph.
	Directed bool
	// Graph ID; or empty if anonymous.
	ID string
	// Graph statements.
	Stmts []Stmt
}

// String returns the string representation of the graph.
func (g *Graph) String() string {
	buf := new(bytes.Buffer)
	if g.Strict {
		buf.WriteString("strict ")
	}
	if g.Directed {
		buf.WriteString("digraph ")
	} else {
		buf.WriteString("graph ")
	}
	if len(g.ID) > 0 {
		fmt.Fprintf(buf, "%s ", g.ID)
	}
	buf.WriteString("{\n")
	for _, stmt := range g.Stmts {
		fmt.Fprintf(buf, "\t%s\n", stmt)
	}
	buf.WriteString("}")
	return buf.String()
}

// === [ Statements ] ==========================================================

// A Stmt represents a statement, and has one of the following underlying types.
//
//    *NodeStmt
//    *EdgeStmt
//    *AttrStmt
//    *Attr
//    *Subgraph
type Stmt interface {
	fmt.Stringer
	// isStmt ensures that only statements can be assigned to the Stmt interface.
	isStmt()
}

// --- [ Node statement ] ------------------------------------------------------

// A NodeStmt represents a node statement.
//
// Examples.
//
//    A [color=blue]
type NodeStmt struct {
	// Node.
	Node *Node
	// Node attributes.
	Attrs []*Attr
}

// String returns the string representation of the node statement.
func (e *NodeStmt) String() string {
	buf := new(bytes.Buffer)
	buf.WriteString(e.Node.String())
	if len(e.Attrs) > 0 {
		buf.WriteString(" [")
		for i, attr := range e.Attrs {
			if i != 0 {
				buf.WriteString(" ")
			}
			buf.WriteString(attr.String())
		}
		buf.WriteString("]")
	}
	return buf.String()
}

// --- [ Edge statement ] ------------------------------------------------------

// An EdgeStmt represents an edge statement.
//
// Examples.
//
//    A -> B
//    A -> {B C}
//    A -> B -> C
type EdgeStmt struct {
	// Source vertex.
	From Vertex
	// Outgoing edge.
	To *Edge
	// Edge attributes.
	Attrs []*Attr
}

// String returns the string representation of the edge statement.
func (e *EdgeStmt) String() string {
	buf := new(bytes.Buffer)
	fmt.Fprintf(buf, "%s %s", e.From, e.To)
	if len(e.Attrs) > 0 {
		buf.WriteString(" [")
		for i, attr := range e.Attrs {
			if i != 0 {
				buf.WriteString(" ")
			}
			buf.WriteString(attr.String())
		}
		buf.WriteString("]")
	}
	return buf.String()
}

// An Edge represents an edge between two vertices.
type Edge struct {
	// Directed edge.
	Directed bool
	// Destination vertex.
	Vertex Vertex
	// Outgoing edge; or nil if none.
	To *Edge
}

// String returns the string representation of the edge.
func (e *Edge) String() string {
	op := "--"
	if e.Directed {
		op = "->"
	}
	if e.To != nil {
		return fmt.Sprintf("%s %s %s", op, e.Vertex, e.To)
	}
	return fmt.Sprintf("%s %s", op, e.Vertex)
}

// --- [ Attribute statement ] -------------------------------------------------

// An AttrStmt represents an attribute statement.
//
// Examples.
//
//    graph [rankdir=LR]
//    node [color=blue fillcolor=red]
//    edge [minlen=1]
type AttrStmt struct {
	// Graph component kind to which the attributes are assigned.
	Kind Kind
	// Attributes.
	Attrs []*Attr
}

// String returns the string representation of the attribute statement.
func (a *AttrStmt) String() string {
	buf := new(bytes.Buffer)
	fmt.Fprintf(buf, "%s [", a.Kind)
	for i, attr := range a.Attrs {
		if i != 0 {
			buf.WriteString(" ")
		}
		buf.WriteString(attr.String())
	}
	buf.WriteString("]")
	return buf.String()
}

// Kind specifies the set of graph components to which attribute statements may
// be assigned.
type Kind uint

// Graph component kinds.
const (
	GraphKind Kind = iota // graph
	NodeKind              // node
	EdgeKind              // edge
)

// String returns the string representation of the graph component kind.
func (k Kind) String() string {
	switch k {
	case GraphKind:
		return "graph"
	case NodeKind:
		return "node"
	case EdgeKind:
		return "edge"
	}
	panic(fmt.Sprintf("invalid graph component kind (%d)", k))
}

// --- [ Attribute ] -----------------------------------------------------------

// An Attr represents an attribute.
//
// Examples.
//
//    rank=same
type Attr struct {
	// Attribute key.
	Key string
	// Attribute value.
	Val string
}

// String returns the string representation of the attribute.
func (a *Attr) String() string {
	return fmt.Sprintf("%s=%s", a.Key, a.Val)
}

// --- [ Subgraph ] ------------------------------------------------------------

// A Subgraph represents a subgraph vertex.
//
// Examples.
//
//    subgraph S {A B C}
type Subgraph struct {
	// Subgraph ID; or empty if none.
	ID string
	// Subgraph statements.
	Stmts []Stmt
}

// String returns the string representation of the subgraph.
func (s *Subgraph) String() string {
	buf := new(bytes.Buffer)
	if len(s.ID) > 0 {
		fmt.Fprintf(buf, "subgraph %s ", s.ID)
	}
	buf.WriteString("{")
	for i, stmt := range s.Stmts {
		if i != 0 {
			buf.WriteString(" ")
		}
		buf.WriteString(stmt.String())
	}
	buf.WriteString("}")
	return buf.String()
}

// isStmt ensures that only statements can be assigned to the Stmt interface.
func (*NodeStmt) isStmt() {}
func (*EdgeStmt) isStmt() {}
func (*AttrStmt) isStmt() {}
func (*Attr) isStmt()     {}
func (*Subgraph) isStmt() {}

// === [ Vertices ] ============================================================

// A Vertex represents a vertex, and has one of the following underlying types.
//
//    *Node
//    *Subgraph
type Vertex interface {
	fmt.Stringer
	// isVertex ensures that only vertices can be assigned to the Vertex
	// interface.
	isVertex()
}

// --- [ Node identifier ] -----------------------------------------------------

// A Node represents a node vertex.
//
// Examples.
//
//    A
//    A:nw
type Node struct {
	// Node ID.
	ID string
	// Node port; or nil if none.
	Port *Port
}

// String returns the string representation of the node.
func (n *Node) String() string {
	if n.Port != nil {
		return fmt.Sprintf("%s%s", n.ID, n.Port)
	}
	return n.ID
}

// A Port specifies where on a node an edge should be aimed.
type Port struct {
	// Port ID; or empty if none.
	ID string
	// Compass point.
	CompassPoint CompassPoint
}

// String returns the string representation of the port.
func (p *Port) String() string {
	buf := new(bytes.Buffer)
	if len(p.ID) > 0 {
		fmt.Fprintf(buf, ":%s", p.ID)
	}
	if p.CompassPoint != CompassPointNone {
		fmt.Fprintf(buf, ":%s", p.CompassPoint)
	}
	return buf.String()
}

// CompassPoint specifies the set of compass points.
type CompassPoint uint

// Compass points.
const (
	CompassPointNone      CompassPoint = iota //
	CompassPointNorth                         // n
	CompassPointNorthEast                     // ne
	CompassPointEast                          // e
	CompassPointSouthEast                     // se
	CompassPointSouth                         // s
	CompassPointSouthWest                     // sw
	CompassPointWest                          // w
	CompassPointNorthWest                     // nw
	CompassPointCenter                        // c
	CompassPointDefault                       // _
)

// String returns the string representation of the compass point.
func (c CompassPoint) String() string {
	switch c {
	case CompassPointNone:
		return ""
	case CompassPointNorth:
		return "n"
	case CompassPointNorthEast:
		return "ne"
	case CompassPointEast:
		return "e"
	case CompassPointSouthEast:
		return "se"
	case CompassPointSouth:
		return "s"
	case CompassPointSouthWest:
		return "sw"
	case CompassPointWest:
		return "w"
	case CompassPointNorthWest:
		return "nw"
	case CompassPointCenter:
		return "c"
	case CompassPointDefault:
		return "_"
	}
	panic(fmt.Sprintf("invalid compass point (%d)", uint(c)))
}

// isVertex ensures that only vertices can be assigned to the Vertex interface.
func (*Node) isVertex()     {}
func (*Subgraph) isVertex() {}
