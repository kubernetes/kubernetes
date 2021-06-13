// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dot

import (
	"fmt"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding"
	"gonum.org/v1/gonum/graph/formats/dot"
	"gonum.org/v1/gonum/graph/formats/dot/ast"
	"gonum.org/v1/gonum/graph/internal/set"
)

// AttributeSetters is implemented by graph values that can set global
// DOT attributes.
type AttributeSetters interface {
	// DOTAttributeSetters returns the global attribute setters.
	DOTAttributeSetters() (graph, node, edge encoding.AttributeSetter)
}

// DOTIDSetter is implemented by types that can set a DOT ID.
type DOTIDSetter interface {
	SetDOTID(id string)
}

// PortSetter is implemented by graph.Edge and graph.Line that can set
// the DOT port and compass directions of an edge.
type PortSetter interface {
	// SetFromPort sets the From port and
	// compass direction of the receiver.
	SetFromPort(port, compass string) error

	// SetToPort sets the To port and compass
	// direction of the receiver.
	SetToPort(port, compass string) error
}

// Unmarshal parses the Graphviz DOT-encoded data and stores the result in dst.
// If the number of graphs encoded in data is not one, an error is returned and
// dst will hold the first graph in data.
//
// Attributes and IDs are unquoted during unmarshalling if appropriate.
func Unmarshal(data []byte, dst encoding.Builder) error {
	file, err := dot.ParseBytes(data)
	if err != nil {
		return err
	}
	err = copyGraph(dst, file.Graphs[0])
	if err == nil && len(file.Graphs) != 1 {
		err = fmt.Errorf("invalid number of graphs; expected 1, got %d", len(file.Graphs))
	}
	return err
}

// UnmarshalMulti parses the Graphviz DOT-encoded data as a multigraph and
// stores the result in dst.
// If the number of graphs encoded in data is not one, an error is returned and
// dst will hold the first graph in data.
//
// Attributes and IDs are unquoted during unmarshalling if appropriate.
func UnmarshalMulti(data []byte, dst encoding.MultiBuilder) error {
	file, err := dot.ParseBytes(data)
	if err != nil {
		return err
	}
	err = copyMultigraph(dst, file.Graphs[0])
	if err == nil && len(file.Graphs) != 1 {
		err = fmt.Errorf("invalid number of graphs; expected 1, got %d", len(file.Graphs))
	}
	return err
}

// copyGraph copies the nodes and edges from the Graphviz AST source graph to
// the destination graph. Edge direction is maintained if present.
func copyGraph(dst encoding.Builder, src *ast.Graph) (err error) {
	defer func() {
		switch e := recover().(type) {
		case nil:
		case error:
			err = e
		default:
			panic(e)
		}
	}()
	gen := &simpleGraph{
		generator: generator{
			directed: src.Directed,
			ids:      make(map[string]graph.Node),
		},
	}
	if dst, ok := dst.(DOTIDSetter); ok {
		dst.SetDOTID(unquoteID(src.ID))
	}
	if a, ok := dst.(AttributeSetters); ok {
		gen.graphAttr, gen.nodeAttr, gen.edgeAttr = a.DOTAttributeSetters()
	}
	for _, stmt := range src.Stmts {
		gen.addStmt(dst, stmt)
	}
	return err
}

// copyMultigraph copies the nodes and edges from the Graphviz AST source graph to
// the destination graph. Edge direction is maintained if present.
func copyMultigraph(dst encoding.MultiBuilder, src *ast.Graph) (err error) {
	defer func() {
		switch e := recover().(type) {
		case nil:
		case error:
			err = e
		default:
			panic(e)
		}
	}()
	gen := &multiGraph{
		generator: generator{
			directed: src.Directed,
			ids:      make(map[string]graph.Node),
		},
	}
	if dst, ok := dst.(DOTIDSetter); ok {
		dst.SetDOTID(unquoteID(src.ID))
	}
	if a, ok := dst.(AttributeSetters); ok {
		gen.graphAttr, gen.nodeAttr, gen.edgeAttr = a.DOTAttributeSetters()
	}
	for _, stmt := range src.Stmts {
		gen.addStmt(dst, stmt)
	}
	return err
}

// A generator keeps track of the information required for generating a Gonum
// graph from a dot AST graph.
type generator struct {
	// Directed graph.
	directed bool
	// Map from dot AST node ID to Gonum node.
	ids map[string]graph.Node
	// Nodes processed within the context of a subgraph, that is to be used as a
	// vertex of an edge.
	subNodes []graph.Node
	// Stack of start indices into the subgraph node slice. The top element
	// corresponds to the start index of the active (or inner-most) subgraph.
	subStart []int
	// graphAttr, nodeAttr and edgeAttr are global graph attributes.
	graphAttr, nodeAttr, edgeAttr encoding.AttributeSetter
}

// node returns the Gonum node corresponding to the given dot AST node ID,
// generating a new such node if none exist.
func (gen *generator) node(dst graph.NodeAdder, id string) graph.Node {
	if n, ok := gen.ids[id]; ok {
		return n
	}
	n := dst.NewNode()
	if n, ok := n.(DOTIDSetter); ok {
		n.SetDOTID(unquoteID(id))
	}
	dst.AddNode(n)
	gen.ids[id] = n
	// Check if within the context of a subgraph, that is to be used as a vertex
	// of an edge.
	if gen.isInSubgraph() {
		// Append node processed within the context of a subgraph, that is to be
		// used as a vertex of an edge
		gen.appendSubgraphNode(n)
	}
	return n
}

type simpleGraph struct{ generator }

// addStmt adds the given statement to the graph.
func (gen *simpleGraph) addStmt(dst encoding.Builder, stmt ast.Stmt) {
	switch stmt := stmt.(type) {
	case *ast.NodeStmt:
		n, ok := gen.node(dst, stmt.Node.ID).(encoding.AttributeSetter)
		if !ok {
			return
		}
		for _, attr := range stmt.Attrs {
			a := encoding.Attribute{
				Key:   unquoteID(attr.Key),
				Value: unquoteID(attr.Val),
			}
			if err := n.SetAttribute(a); err != nil {
				panic(fmt.Errorf("unable to unmarshal node DOT attribute (%s=%s): %v", a.Key, a.Value, err))
			}
		}
	case *ast.EdgeStmt:
		gen.addEdgeStmt(dst, stmt)
	case *ast.AttrStmt:
		var n encoding.AttributeSetter
		var dst string
		switch stmt.Kind {
		case ast.GraphKind:
			if gen.graphAttr == nil {
				return
			}
			n = gen.graphAttr
			dst = "graph"
		case ast.NodeKind:
			if gen.nodeAttr == nil {
				return
			}
			n = gen.nodeAttr
			dst = "node"
		case ast.EdgeKind:
			if gen.edgeAttr == nil {
				return
			}
			n = gen.edgeAttr
			dst = "edge"
		default:
			panic("unreachable")
		}
		for _, attr := range stmt.Attrs {
			a := encoding.Attribute{
				Key:   unquoteID(attr.Key),
				Value: unquoteID(attr.Val),
			}
			if err := n.SetAttribute(a); err != nil {
				panic(fmt.Errorf("unable to unmarshal global %s DOT attribute (%s=%s): %v", dst, a.Key, a.Value, err))
			}
		}
	case *ast.Attr:
		// ignore.
	case *ast.Subgraph:
		for _, stmt := range stmt.Stmts {
			gen.addStmt(dst, stmt)
		}
	default:
		panic(fmt.Sprintf("unknown statement type %T", stmt))
	}
}

// basicEdge is an edge without the Reverse method to
// allow satisfaction by both graph.Edge and graph.Line.
type basicEdge interface {
	From() graph.Node
	To() graph.Node
}

// applyPortsToEdge applies the available port metadata from an ast.Edge
// to a graph.Edge
func applyPortsToEdge(from ast.Vertex, to *ast.Edge, edge basicEdge) {
	if ps, isPortSetter := edge.(PortSetter); isPortSetter {
		if n, vertexIsNode := from.(*ast.Node); vertexIsNode {
			if n.Port != nil {
				err := ps.SetFromPort(unquoteID(n.Port.ID), n.Port.CompassPoint.String())
				if err != nil {
					panic(fmt.Errorf("unable to unmarshal edge port (:%s:%s)", n.Port.ID, n.Port.CompassPoint.String()))
				}
			}
		}

		if n, vertexIsNode := to.Vertex.(*ast.Node); vertexIsNode {
			if n.Port != nil {
				err := ps.SetToPort(unquoteID(n.Port.ID), n.Port.CompassPoint.String())
				if err != nil {
					panic(fmt.Errorf("unable to unmarshal edge DOT port (:%s:%s)", n.Port.ID, n.Port.CompassPoint.String()))
				}
			}
		}
	}
}

// addEdgeStmt adds the given edge statement to the graph.
func (gen *simpleGraph) addEdgeStmt(dst encoding.Builder, stmt *ast.EdgeStmt) {
	fs := gen.addVertex(dst, stmt.From)
	ts := gen.addEdge(dst, stmt.To, stmt.Attrs)
	defer func() {
		switch e := recover().(type) {
		case nil:
			// Do nothing.
		case error:
			panic(e)
		default:
			panic(fmt.Errorf("panic setting edge: %v", e))
		}
	}()
	for _, f := range fs {
		for _, t := range ts {
			edge := dst.NewEdge(f, t)
			dst.SetEdge(edge)
			applyPortsToEdge(stmt.From, stmt.To, edge)
			addEdgeAttrs(edge, stmt.Attrs)
		}
	}
}

// addVertex adds the given vertex to the graph, and returns its set of nodes.
func (gen *simpleGraph) addVertex(dst encoding.Builder, v ast.Vertex) []graph.Node {
	switch v := v.(type) {
	case *ast.Node:
		n := gen.node(dst, v.ID)
		return []graph.Node{n}
	case *ast.Subgraph:
		gen.pushSubgraph()
		for _, stmt := range v.Stmts {
			gen.addStmt(dst, stmt)
		}
		return gen.popSubgraph()
	default:
		panic(fmt.Sprintf("unknown vertex type %T", v))
	}
}

// addEdge adds the given edge to the graph, and returns its set of nodes.
func (gen *simpleGraph) addEdge(dst encoding.Builder, to *ast.Edge, attrs []*ast.Attr) []graph.Node {
	if !gen.directed && to.Directed {
		panic(fmt.Errorf("directed edge to %v in undirected graph", to.Vertex))
	}
	fs := gen.addVertex(dst, to.Vertex)
	if to.To != nil {
		ts := gen.addEdge(dst, to.To, attrs)
		for _, f := range fs {
			for _, t := range ts {
				edge := dst.NewEdge(f, t)
				dst.SetEdge(edge)
				applyPortsToEdge(to.Vertex, to.To, edge)
				addEdgeAttrs(edge, attrs)
			}
		}
	}
	return fs
}

// pushSubgraph pushes the node start index of the active subgraph onto the
// stack.
func (gen *generator) pushSubgraph() {
	gen.subStart = append(gen.subStart, len(gen.subNodes))
}

// popSubgraph pops the node start index of the active subgraph from the stack,
// and returns the nodes processed since.
func (gen *generator) popSubgraph() []graph.Node {
	// Get nodes processed since the subgraph became active.
	start := gen.subStart[len(gen.subStart)-1]
	// TODO: Figure out a better way to store subgraph nodes, so that duplicates
	// may not occur.
	nodes := unique(gen.subNodes[start:])
	// Remove subgraph from stack.
	gen.subStart = gen.subStart[:len(gen.subStart)-1]
	if len(gen.subStart) == 0 {
		// Remove subgraph nodes when the bottom-most subgraph has been processed.
		gen.subNodes = gen.subNodes[:0]
	}
	return nodes
}

// unique returns the set of unique nodes contained within ns.
func unique(ns []graph.Node) []graph.Node {
	var nodes []graph.Node
	seen := make(set.Int64s)
	for _, n := range ns {
		id := n.ID()
		if seen.Has(id) {
			// skip duplicate node
			continue
		}
		seen.Add(id)
		nodes = append(nodes, n)
	}
	return nodes
}

// isInSubgraph reports whether the active context is within a subgraph, that is
// to be used as a vertex of an edge.
func (gen *generator) isInSubgraph() bool {
	return len(gen.subStart) > 0
}

// appendSubgraphNode appends the given node to the slice of nodes processed
// within the context of a subgraph.
func (gen *generator) appendSubgraphNode(n graph.Node) {
	gen.subNodes = append(gen.subNodes, n)
}

type multiGraph struct{ generator }

// addStmt adds the given statement to the multigraph.
func (gen *multiGraph) addStmt(dst encoding.MultiBuilder, stmt ast.Stmt) {
	switch stmt := stmt.(type) {
	case *ast.NodeStmt:
		n, ok := gen.node(dst, stmt.Node.ID).(encoding.AttributeSetter)
		if !ok {
			return
		}
		for _, attr := range stmt.Attrs {
			a := encoding.Attribute{
				Key:   unquoteID(attr.Key),
				Value: unquoteID(attr.Val),
			}
			if err := n.SetAttribute(a); err != nil {
				panic(fmt.Errorf("unable to unmarshal node DOT attribute (%s=%s): %v", a.Key, a.Value, err))
			}
		}
	case *ast.EdgeStmt:
		gen.addEdgeStmt(dst, stmt)
	case *ast.AttrStmt:
		var n encoding.AttributeSetter
		var dst string
		switch stmt.Kind {
		case ast.GraphKind:
			if gen.graphAttr == nil {
				return
			}
			n = gen.graphAttr
			dst = "graph"
		case ast.NodeKind:
			if gen.nodeAttr == nil {
				return
			}
			n = gen.nodeAttr
			dst = "node"
		case ast.EdgeKind:
			if gen.edgeAttr == nil {
				return
			}
			n = gen.edgeAttr
			dst = "edge"
		default:
			panic("unreachable")
		}
		for _, attr := range stmt.Attrs {
			a := encoding.Attribute{
				Key:   unquoteID(attr.Key),
				Value: unquoteID(attr.Val),
			}
			if err := n.SetAttribute(a); err != nil {
				panic(fmt.Errorf("unable to unmarshal global %s DOT attribute (%s=%s): %v", dst, a.Key, a.Value, err))
			}
		}
	case *ast.Attr:
		// ignore.
	case *ast.Subgraph:
		for _, stmt := range stmt.Stmts {
			gen.addStmt(dst, stmt)
		}
	default:
		panic(fmt.Sprintf("unknown statement type %T", stmt))
	}
}

// addEdgeStmt adds the given edge statement to the multigraph.
func (gen *multiGraph) addEdgeStmt(dst encoding.MultiBuilder, stmt *ast.EdgeStmt) {
	fs := gen.addVertex(dst, stmt.From)
	ts := gen.addLine(dst, stmt.To, stmt.Attrs)
	for _, f := range fs {
		for _, t := range ts {
			edge := dst.NewLine(f, t)
			dst.SetLine(edge)
			applyPortsToEdge(stmt.From, stmt.To, edge)
			addEdgeAttrs(edge, stmt.Attrs)
		}
	}
}

// addVertex adds the given vertex to the multigraph, and returns its set of nodes.
func (gen *multiGraph) addVertex(dst encoding.MultiBuilder, v ast.Vertex) []graph.Node {
	switch v := v.(type) {
	case *ast.Node:
		n := gen.node(dst, v.ID)
		return []graph.Node{n}
	case *ast.Subgraph:
		gen.pushSubgraph()
		for _, stmt := range v.Stmts {
			gen.addStmt(dst, stmt)
		}
		return gen.popSubgraph()
	default:
		panic(fmt.Sprintf("unknown vertex type %T", v))
	}
}

// addLine adds the given edge to the multigraph, and returns its set of nodes.
func (gen *multiGraph) addLine(dst encoding.MultiBuilder, to *ast.Edge, attrs []*ast.Attr) []graph.Node {
	if !gen.directed && to.Directed {
		panic(fmt.Errorf("directed edge to %v in undirected graph", to.Vertex))
	}
	fs := gen.addVertex(dst, to.Vertex)
	if to.To != nil {
		ts := gen.addLine(dst, to.To, attrs)
		for _, f := range fs {
			for _, t := range ts {
				edge := dst.NewLine(f, t)
				dst.SetLine(edge)
				applyPortsToEdge(to.Vertex, to.To, edge)
				addEdgeAttrs(edge, attrs)
			}
		}
	}
	return fs
}

// addEdgeAttrs adds the attributes to the given edge.
func addEdgeAttrs(edge basicEdge, attrs []*ast.Attr) {
	e, ok := edge.(encoding.AttributeSetter)
	if !ok {
		return
	}
	for _, attr := range attrs {
		a := encoding.Attribute{
			Key:   unquoteID(attr.Key),
			Value: unquoteID(attr.Val),
		}
		if err := e.SetAttribute(a); err != nil {
			panic(fmt.Errorf("unable to unmarshal edge DOT attribute (%s=%s): %v", a.Key, a.Value, err))
		}
	}
}

// unquoteID unquotes the given string if needed in the context of an ID. If s
// is not already quoted the original string is returned.
func unquoteID(s string) string {
	// To make round-trips idempotent, don't unquote quoted HTML-like strings
	//
	//    /^"<.*>"$/
	if len(s) >= 4 && strings.HasPrefix(s, `"<`) && strings.HasSuffix(s, `>"`) {
		return s
	}
	// Unquote quoted string if possible.
	if t, err := strconv.Unquote(s); err == nil {
		return t
	}
	// On error, either s is not quoted or s is quoted but contains invalid
	// characters, in both cases we return the original string rather than
	// panicking.
	return s
}
